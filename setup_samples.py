"""Generate sample text files for each document class.

Creates (if missing):
  - class_examples/invoice
  - class_examples/bank_statement
  - class_examples/resume
  - class_examples/ITR
  - class_examples/government_id

For each folder, writes exactly 15 files: ex1.txt .. ex15.txt
Each file has 2–5 lines of representative, slightly varied content.
"""

import os
from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parent
CLASS_DIR = BASE_DIR / "class_examples"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_files(target_dir: Path, documents: List[List[str]]) -> None:
    # Ensure exactly 15 files (ex1.txt..ex15.txt). If more templates provided, cycle.
    ensure_dir(target_dir)
    total = 15
    for i in range(1, total + 1):
        lines = documents[(i - 1) % len(documents)]
        # Light variation knobs
        amount = 100 + i * 7
        alt_amount = 250 + i * 13
        year = 2020 + (i % 6)
        seq = f"{i:03d}"
        varied = [
            l.replace("{AMOUNT}", str(amount))
             .replace("{ALT_AMOUNT}", str(alt_amount))
             .replace("{YEAR}", str(year))
             .replace("{SEQ}", seq)
        for l in lines]
        fn = target_dir / f"ex{i}.txt"
        with fn.open("w", encoding="utf-8") as f:
            f.write("\n".join(varied).strip() + "\n")


def invoice_templates() -> List[List[str]]:
    return [
        [
            "Invoice No: INV-{SEQ}",
            "Bill To: Acme Corp",
            "GST: 27ABCDE1234F1Z5",
            "Subtotal: {AMOUNT}.00 | Amount Due: {ALT_AMOUNT}.50",
            "Payment Terms: Net 30 | Authorized Signatory",
        ],
        [
            "INVOICE #{SEQ}",
            "BILL TO: John Doe Pvt Ltd",
            "Amount Due: {AMOUNT}.75 | Subtotal: {ALT_AMOUNT}.00",
            "GST Included | Payment Terms: Due on Receipt",
        ],
        [
            "Tax Invoice - Series A {SEQ}",
            "Bill To: Globex LLC",
            "Subtotal {AMOUNT}.25 | GST 18%",
            "Authorized Signatory: Finance Manager",
        ],
        [
            "Invoice Number: 2024-{SEQ}",
            "Amount Due: {ALT_AMOUNT}.00",
            "Payment Terms: Net 15 | Subtotal: {AMOUNT}.00",
        ],
        [
            "Commercial Invoice INV-{SEQ}",
            "Bill To: Example Industries",
            "Subtotal {AMOUNT}.00 | GST Registered",
            "Authorized Signatory",
        ],
    ]


def bank_statement_templates() -> List[List[str]]:
    return [
        [
            "Bank Statement - {YEAR}",
            "Account Number: 00123456{SEQ}",
            "Transaction: Debit {AMOUNT}.00 | Credit {ALT_AMOUNT}.00",
            "Closing Balance: {ALT_AMOUNT}.25 | NEFT Ref: NEFT{SEQ}",
        ],
        [
            "Account Number 99{SEQ}77665",
            "Transaction: Credit {AMOUNT}.50",
            "Balance: {ALT_AMOUNT}.10 | NEFT UTR: ABC{SEQ}XYZ",
            "Debit Charges: 0.00 | Closing Balance",
        ],
        [
            "BANK STATEMENT",
            "Account Number: 12345{SEQ}0",
            "Debit {AMOUNT}.90 | Credit {ALT_AMOUNT}.40 | Balance {AMOUNT}.15",
        ],
        [
            "Monthly Statement {YEAR}-{SEQ}",
            "NEFT Received | Transaction: Credit {ALT_AMOUNT}.00",
            "Closing Balance: {AMOUNT}.00",
        ],
        [
            "Statement of Account",
            "Account Number: 55{SEQ}332211",
            "Debit {AMOUNT}.60 | Credit {AMOUNT}.75 | Closing Balance",
        ],
    ]


def resume_templates() -> List[List[str]]:
    return [
        [
            "Resume - Candidate {SEQ}",
            "Experience: 3+ years in Software Development",
            "Education: B.Tech | Skills: Python, ML, SQL",
            "Projects: Analytics Platform | Contact: candidate{SEQ}@mail.com",
            "LinkedIn: linkedin.com/in/candidate{SEQ} | Achievements: Employee of the Year",
        ],
        [
            "RESUME",
            "Experience: Data Analyst | Education: MSc Statistics",
            "Skills: Pandas, Visualization | Projects: Dashboard {YEAR}",
            "Contact: +91-90000{SEQ} | LinkedIn provided",
        ],
        [
            "Professional Resume",
            "Education: BE Computer Science | Experience: Intern + 1 yr",
            "Skills: APIs, JavaScript, Python | Achievements: Hackathon Winner",
        ],
        [
            "Resume Summary",
            "Projects: NLP Pipeline {YEAR} | Skills: Transformers, FastAPI",
            "Contact: resume{SEQ}@example.com | LinkedIn included",
        ],
        [
            "Curriculum Vitae",
            "Experience: QA Engineer | Education: B.Sc",
            "Skills: Selenium, JUnit | Achievements: Test Coverage 95%",
        ],
    ]


def itr_templates() -> List[List[str]]:
    return [
        [
            "Income Tax Return (ITR) AY {YEAR}-{YEAR+1}",
            "PAN: ABCDE{SEQ}F",
            "Gross Total Income: {ALT_AMOUNT}00",
            "TDS Claimed | 80C Deduction | Refund Status: Initiated",
        ],
        [
            "ITR Acknowledgement",
            "Assessment Year: {YEAR}-{YEAR+1} | PAN: PAN{SEQ}XYZ",
            "Gross Total Income {AMOUNT}00 | 80C Deduction Applied",
        ],
        [
            "Income Tax Return Filing",
            "PAN Provided | TDS Details | Refund Requested",
            "Gross Total Income {ALT_AMOUNT}50",
        ],
        [
            "ITR Verification",
            "Assessment Year {YEAR}-{YEAR+1}",
            "80C Deduction | PAN AAAPL{SEQ}Z | Refund Credited",
        ],
        [
            "ITR Form-1",
            "PAN: AAAA{SEQ}A | Gross Total Income {AMOUNT}00",
            "TDS and 80C Deduction considered | Refund Pending",
        ],
    ]


def government_id_templates() -> List[List[str]]:
    return [
        [
            "Government of India",
            "ID Number: GOV{SEQ}9900",
            "DOB: 1990-01-{(int('1') if False else '')}",
            "Issued by: Regional Authority | Passport Mentioned | Voter ID present",
        ],
        [
            "Govt. of India ID Card",
            "DOB: 198{(0)}-07-15 | ID Number: ID{SEQ}22",
            "Issued by Authority | Driving License Reference",
        ],
        [
            "Identity Document",
            "ID Number: X{SEQ}YZ7788 | DOB: 1995-12-05",
            "Passport Reference | PAN Card Mentioned",
        ],
        [
            "Government ID Details",
            "Issued by Central Authority | Voter ID Linked",
            "DOB: 2000-02-20 | ID Number: ID{SEQ}",
        ],
        [
            "Official ID Proof",
            "PAN Card Linked | Driving License Available",
            "Government of India Seal | ID Number: GID{SEQ}",
        ],
    ]


def main() -> None:
    classes: Dict[str, List[List[str]]]= {
        "invoice": invoice_templates(),
        "bank_statement": bank_statement_templates(),
        "resume": resume_templates(),
        "ITR": itr_templates(),
        "government_id": government_id_templates(),
    }

    for cls, templates in classes.items():
        target = CLASS_DIR / cls
        write_files(target, templates)

    print("✅ Sample folders populated with 15 snippets each")


if __name__ == "__main__":
    main()


