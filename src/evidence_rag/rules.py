import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .models import (
    Answer,
    AnswerMetrics,
    ApprovalRequest,
    Decision,
    SearchResult,
)

MONEY = re.compile(r"\$(?P<amount>\d+(?:\.\d{1,2})?)")


def format_money(amount: Decimal) -> str:
    return f"${amount:,.2f}".removesuffix(".00")


@dataclass(frozen=True, slots=True)
class RefundApprovalPolicy:
    agent_limit: Decimal
    manager_role: str
    citation_id: str


class PolicyRuleEngine:
    """Apply deterministic business rules before asking a generative model."""

    def __init__(self, refund_policy: RefundApprovalPolicy) -> None:
        self.refund_policy = refund_policy

    @classmethod
    def from_path(cls, path: Path) -> "PolicyRuleEngine":
        payload = json.loads(path.read_text(encoding="utf-8"))
        refund = payload["refund_approval"]
        return cls(
            RefundApprovalPolicy(
                agent_limit=Decimal(refund["agent_limit"]),
                manager_role=refund["manager_role"],
                citation_id=refund["citation_id"],
            )
        )

    def answer(self, question: str, evidence: list[SearchResult]) -> Answer | None:
        question_lower = question.lower()
        is_travel_approval = (
            "travel agent" in question_lower
            and "reimbursement" in question_lower
            and any(word in question_lower for word in ("approve", "authority", "authorize"))
        )

        travel_evidence = next(
            (
                result
                for result in evidence
                if result.chunk.id == "handbook#5"
                and "travel agents cannot approve reimbursements" in result.chunk.text.lower()
            ),
            None,
        )

        if is_travel_approval and travel_evidence is not None:
            return Answer(
                text="No. Travel agents cannot approve reimbursements.",
                citations=(travel_evidence.chunk.id,),
                abstained=False,
                decision="no",
                metrics=AnswerMetrics(
                    provider="rule-engine",
                    model="travel-approval-v1",
                ),
            )

        role_mentioned = any(
            phrase in question_lower for phrase in ("support agent", "agent’s", "agent's")
        )
        action_mentioned = any(word in question_lower for word in ("approve", "authority"))
        topic_mentioned = any(word in question_lower for word in ("refund", "reimbursement"))

        is_refund_decision = role_mentioned and action_mentioned and topic_mentioned
        amount_match = MONEY.search(question)
        allowed_citations = {result.chunk.id for result in evidence}
        policy = self.refund_policy
        if (
            not is_refund_decision
            or amount_match is None
            or policy.citation_id not in allowed_citations
        ):
            return None

        requested = Decimal(amount_match.group("amount"))
        request = ApprovalRequest(
            role="support_agent",
            action="approve_refund",
            amount=requested,
        )
        decision = self.decide_support_refund(request)
        if decision is None:
            return None

        approved = decision == "yes"

        if approved:
            text = (
                f"Yes. A support agent may approve a {format_money(requested)} refund because it "
                f"does not exceed the {format_money(policy.agent_limit)} agent limit."
            )
        else:
            text = (
                f"No. A support agent cannot independently approve a {format_money(requested)} "
                f"refund; approval from a {policy.manager_role} is required because the agent "
                f"limit is {format_money(policy.agent_limit)}."
            )
        return Answer(
            text=text,
            citations=(policy.citation_id,),
            decision="yes" if approved else "no",
            metrics=AnswerMetrics(provider="rule-engine", model="refund-approval-v1"),
        )

    def decide_support_refund(self, request: ApprovalRequest) -> Decision | None:
        if (
            request.role != "support_agent"
            or request.action != "approve_refund"
            or request.amount is None
        ):
            return None

        if request.amount <= self.refund_policy.agent_limit:
            return "yes"

        return "no"

    def answer_request(
        self,
        request: ApprovalRequest,
        evidence: list[SearchResult],
    ) -> Answer | None:
        if request.action != "approve_refund":
            return None

        if request.intent == "find_approver":
            policy = self.refund_policy
            allowed_citations = {result.chunk.id for result in evidence}

            if (
                request.amount is None
                or request.role not in (None, "support_agent")
                or policy.citation_id not in allowed_citations
            ):
                return None

            amount = format_money(request.amount)
            limit = format_money(policy.agent_limit)

            if request.amount > policy.agent_limit:
                text = (
                    f"A {policy.manager_role} must approve a "
                    f"{amount} refund because it exceeds "
                    f"the {limit} support-agent limit."
                )
            else:
                text = (
                    f"A support agent may approve a {amount} refund "
                    f"because it does not exceed the {limit} limit."
                )

            return Answer(
                text=text,
                citations=(policy.citation_id,),
                abstained=False,
                decision="not_applicable",
                metrics=AnswerMetrics(
                    provider="rule-engine",
                    model="refund-approval-v1",
                ),
            )

        if request.intent != "check_authority":
            return None

        if request.role is None:
            return Answer(
                text="What type of agent do you mean?",
                citations=(),
                abstained=False,
                decision="not_applicable",
                needs_clarification=True,
                metrics=AnswerMetrics(
                    provider="rule-engine",
                    model="refund-approval-v1",
                ),
            )

        if request.role == "travel_agent" and request.action == "approve_refund":
            travel_evidence = next(
                (
                    result
                    for result in evidence
                    if result.chunk.id == "handbook#5"
                    and "travel agents cannot approve reimbursements" in result.chunk.text.lower()
                ),
                None,
            )

            if travel_evidence is None:
                return None

            return Answer(
                text="No. Travel agents cannot approve reimbursements.",
                citations=(travel_evidence.chunk.id,),
                abstained=False,
                decision="no",
                metrics=AnswerMetrics(
                    provider="rule-engine",
                    model="travel-approval-v1",
                ),
            )

        policy = self.refund_policy
        allowed_citations = {result.chunk.id for result in evidence}

        if policy.citation_id not in allowed_citations:
            return None

        if request.role == "support_agent" and request.amount is None:
            return Answer(
                text="What is the refund amount?",
                citations=(),
                abstained=False,
                decision="not_applicable",
                needs_clarification=True,
                metrics=AnswerMetrics(
                    provider="rule-engine",
                    model="refund-approval-v1",
                ),
            )

        decision = self.decide_support_refund(request)
        if decision is None or request.amount is None:
            return None

        amount = format_money(request.amount)
        limit = format_money(policy.agent_limit)

        if decision == "yes":
            text = (
                f"Yes. A support agent may approve a {amount} refund "
                f"because it does not exceed the {limit} agent limit."
            )
        else:
            text = (
                f"No. A support agent cannot independently approve "
                f"a {amount} refund; approval from a "
                f"{policy.manager_role} is required because "
                f"the agent limit is {limit}."
            )

        return Answer(
            text=text,
            citations=(policy.citation_id,),
            abstained=False,
            decision=decision,
            metrics=AnswerMetrics(
                provider="rule-engine",
                model="refund-approval-v1",
            ),
        )
