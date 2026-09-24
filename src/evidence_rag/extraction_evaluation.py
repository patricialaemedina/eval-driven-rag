from .models import ApprovalRequest


def evaluate_extraction(
    actual: ApprovalRequest,
    expected: ApprovalRequest,
) -> dict[str, bool]:
    checks = {
        "role_correct": actual.role == expected.role,
        "action_correct": actual.action == expected.action,
        "amount_correct": actual.amount == expected.amount,
        "intent_correct": actual.intent == expected.intent,
    }

    return {
        **checks,
        "passed": all(checks.values()),
    }
