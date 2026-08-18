"""Generate seed dataset for the Auto Email Ticket Categorizer project.

Output file: tickets.csv
Required columns: subject, body, category
Categories: Billing, Technical, HR, General
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

OUTPUT_FILE = Path("tickets.csv")
FIELDNAMES = ["subject", "body", "category"]


def build_seed_rows() -> List[Dict[str, str]]:
    """Create a balanced labeled dataset for initial model training/testing."""
    return [
        # Billing (6)
        {
            "subject": "Charged twice for subscription",
            "body": "I see two card charges for the same monthly plan. Please refund one.",
            "category": "Billing",
        },
        {
            "subject": "Invoice amount is incorrect",
            "body": "My latest invoice shows a higher amount than the quoted annual renewal.",
            "category": "Billing",
        },
        {
            "subject": "Need GST invoice copy",
            "body": "Can you share the tax invoice for payment done on Monday?",
            "category": "Billing",
        },
        {
            "subject": "Refund not received",
            "body": "Refund was promised last week but still not credited to my bank account.",
            "category": "Billing",
        },
        {
            "subject": "Payment failed but money debited",
            "body": "Checkout failed but amount got debited. Please verify and reverse.",
            "category": "Billing",
        },
        {
            "subject": "Apply promo credit",
            "body": "My discount coupon was not applied on the final bill.",
            "category": "Billing",
        },
        # Technical (6)
        {
            "subject": "Unable to login",
            "body": "Password reset link expires instantly and I cannot access dashboard.",
            "category": "Technical",
        },
        {
            "subject": "Production server down",
            "body": "The app is down for all users and API requests are failing urgently.",
            "category": "Technical",
        },
        {
            "subject": "Mobile app crashes",
            "body": "Android app closes on startup after the latest update.",
            "category": "Technical",
        },
        {
            "subject": "2FA code not working",
            "body": "OTP is accepted on web but rejected on desktop client.",
            "category": "Technical",
        },
        {
            "subject": "Email notifications stopped",
            "body": "No alert emails are being sent from the system since yesterday.",
            "category": "Technical",
        },
        {
            "subject": "File upload timeout",
            "body": "Large PDF uploads fail with timeout error and retry loop.",
            "category": "Technical",
        },
        # HR (6)
        {
            "subject": "Leave balance confirmation",
            "body": "Please confirm my remaining paid leave days this quarter.",
            "category": "HR",
        },
        {
            "subject": "Payroll slip missing",
            "body": "My salary slip for this month is not visible in the employee portal.",
            "category": "HR",
        },
        {
            "subject": "Interview reschedule request",
            "body": "Can we move candidate interview to Friday afternoon?",
            "category": "HR",
        },
        {
            "subject": "Update emergency contact",
            "body": "I need to change my emergency contact details in records.",
            "category": "HR",
        },
        {
            "subject": "Question about maternity policy",
            "body": "Could you share the latest maternity leave and benefit policy details?",
            "category": "HR",
        },
        {
            "subject": "Onboarding documents pending",
            "body": "New joiner has not received onboarding checklist and forms.",
            "category": "HR",
        },
        # General (6)
        {
            "subject": "How to change profile photo",
            "body": "Please guide me to update account profile picture settings.",
            "category": "General",
        },
        {
            "subject": "Need product brochure",
            "body": "Can you share an overview document of available plans and features?",
            "category": "General",
        },
        {
            "subject": "Office location details",
            "body": "What is the address and parking information for the Bangalore office?",
            "category": "General",
        },
        {
            "subject": "Request callback",
            "body": "Please arrange a support callback in the afternoon.",
            "category": "General",
        },
        {
            "subject": "Feature availability question",
            "body": "Is dark mode available for all users or only enterprise plans?",
            "category": "General",
        },
        {
            "subject": "Where can I find user handbook",
            "body": "I need the link to documentation and getting started guide.",
            "category": "General",
        },
    ]


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Write seed rows to CSV for model training pipeline."""
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_seed_rows()
    write_csv(rows, OUTPUT_FILE)
    print(f"Created dataset: {OUTPUT_FILE}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    main()
