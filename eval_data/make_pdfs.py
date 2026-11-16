"""Build the two PDF documents of the evaluation corpus.

    uv run --with reportlab python eval_data/make_pdfs.py

They look like real company PDFs on purpose: a running header and footer with the page number on every page,
numbered headings instead of Markdown ones, a contents page and tables whose rows have no punctuation. Those are
the things that make PDFs harder to ingest than Markdown. The PDFs are committed, so this only runs when their
text changes.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CORPUS = Path(__file__).resolve().parent / "corpus"
STYLES = getSampleStyleSheet()

BUDGET_CODES = [
    "Cloud hosting | BC-4410 | Platform team", "Software licences | BC-4420 | IT",
    "Hardware maintenance | BC-4430 | IT", "Office rent | BC-5110 | Facilities",
    "Cleaning services | BC-5120 | Facilities", "Office supplies | BC-5130 | Facilities",
    "Catering for events | BC-5140 | People team", "Recruitment agencies | BC-6110 | People team",
    "Job advertising | BC-6120 | People team", "Training providers | BC-6130 | People team",
    "Legal advice | BC-7110 | Legal", "Audit fees | BC-7120 | Finance", "Tax advice | BC-7130 | Finance",
    "Insurance | BC-7140 | Finance", "Payment network fees | BC-8110 | Payments operations",
    "Card scheme fees | BC-8120 | Payments operations", "Fraud screening services | BC-8130 | Risk",
    "Identity verification | BC-8140 | Risk", "Translation services | BC-9110 | Marketing",
    "Advertising | BC-9120 | Marketing", "Events and sponsorship | BC-9130 | Marketing",
    "Market research | BC-9140 | Marketing", "Design agencies | BC-9150 | Marketing",
    "Customer support outsourcing | BC-9210 | Support", "Telephony | BC-9220 | Support",
    "Consultancy | BC-9310 | Finance director", "Security testing | BC-9320 | Security",
    "Data providers | BC-9330 | Data team", "Couriers and postage | BC-9410 | Facilities",
    "Travel agency fees | BC-9420 | Finance",
]

PROCUREMENT = {
    "file": "procurement-policy.pdf",
    "title": "Procurement Policy",
    "contents": ["1. Scope", "2. Purchase thresholds", "3. Purchase orders", "4. Approved suppliers",
                 "5. Contracts and renewals", "6. Payment terms", "7. Conflicts of interest", "8. Exceptions",
                 "9. Budget codes"],
    "sections": [
        ("1. Scope", [
            "This policy applies whenever Northwind Payments buys goods or services from another company. It does "
            "not cover expenses that employees pay themselves and claim back, which follow the expenses policy, "
            "and it does not cover laptops and other IT equipment, which are ordered through the IT equipment "
            "policy.",
            "The goal is simple: spend company money on purpose, get a fair price, and keep a record that an "
            "auditor can follow from the request to the payment.",
        ]),
        ("2. Purchase thresholds", [
            "The number of quotes you need and who approves the purchase depend on the total value of the "
            "purchase over the whole contract, including taxes. Splitting one purchase into smaller orders to stay "
            "under a threshold is a breach of this policy.",
            ("table", [["Total value (EUR)", "Quotes needed", "Approver"],
                       ["Up to 1,000", "1 quote", "Team lead"],
                       ["1,001 to 10,000", "2 quotes", "Department head"],
                       ["10,001 to 50,000", "3 quotes", "Finance director"],
                       ["Above 50,000", "Formal tender", "Executive committee"]]),
            "Quotes must be written and dated, and they must describe the same scope so that they can be compared.",
        ]),
        ("3. Purchase orders", [
            "Every purchase above 1,000 EUR needs a purchase order raised in Buyflow before the supplier starts "
            "work. Purchase order numbers start with PO- followed by six digits. Give the number to the supplier "
            "and ask them to quote it on every invoice.",
            "Invoices that arrive without a purchase order number are returned to the supplier with code PRC-41 "
            "and are not paid until they are sent again with the number.",
        ]),
        ("4. Approved suppliers", [
            "Buy from a supplier on the approved supplier list whenever one can do the work. Adding a new supplier "
            "to the approved list takes 10 working days, so start early. The procurement team checks the company "
            "registration, bank details and sanctions lists for every new supplier.",
            "When the supplier will process customer data, it must also complete security questionnaire SEC-Q2 "
            "before the contract is signed. The approved supplier list is reviewed every 12 months and suppliers "
            "that were not used in that time are removed.",
        ]),
        ("5. Contracts and renewals", [
            "Contracts longer than 24 months, and every contract that renews automatically, need a review by the "
            "legal team before signature. The contract owner sets a reminder in Buyflow 90 days before the renewal "
            "date so that there is time to renegotiate or cancel.",
        ]),
        ("6. Payment terms", [
            "Standard payment terms are 30 days from the invoice date. Shorter terms need the approval of the "
            "finance director. An early payment discount is only worth taking when it is at least 2 percent.",
        ]),
        ("7. Conflicts of interest", [
            "If you have a personal relationship with a supplier, for example a relative who works there or a "
            "shareholding, declare it with form PRC-07 before the supplier is selected, and do not take part in "
            "the selection. Gifts from suppliers worth more than 50 EUR must be declined.",
        ]),
        ("8. Exceptions", [
            "A purchase from a single supplier without competing quotes is a sole-source purchase. It needs a "
            "written justification explaining why no other supplier can do the work, approved by the finance "
            "director. Emergencies that stop customer payments may be bought first and documented within 5 "
            "working days.",
        ]),
        ("9. Budget codes", [
            "Put the budget code of the spending category on every purchase order. The owner of the code checks "
            "the budget before the order is approved.",
            ("table", [["Spending category", "Budget code", "Owner"],
                       *[row.split(" | ") for row in BUDGET_CODES]]),
            "If no category fits, ask the finance team for a code before raising the purchase order.",
        ]),
    ],
}

MOBILITY = {
    "file": "mobility-programme.pdf",
    "title": "Mobility Programme",
    "contents": ["1. Who can join", "2. Monthly transport allowance", "3. Company bikes", "4. Parking and charging",
                 "5. Changing your choice", "6. Leaving the company"],
    "sections": [
        ("1. Who can join", [
            "Every employee with a contract longer than 6 months can join the mobility programme from their first "
            "day. The programme is a benefit paid through payroll. It is separate from expense claims: commuting "
            "between home and your usual office is never claimed as an expense.",
        ]),
        ("2. Monthly transport allowance", [
            "The allowance depends on the office in your contract. In Barcelona and Madrid it is loaded every month "
            "onto a personal public transport card. In Lisbon, and for remote contracts, it is paid as a top-up to "
            "your salary.",
            ("table", [["Office", "Monthly allowance", "Paid as"],
                       ["Barcelona", "60 EUR", "Transport card"],
                       ["Madrid", "55 EUR", "Transport card"],
                       ["Lisbon", "45 EUR", "Salary top-up"],
                       ["Remote contract", "20 EUR", "Salary top-up"]]),
            "The allowance is not paid for months in which you are on unpaid leave for the whole month.",
        ]),
        ("3. Company bikes", [
            "You can lease a bike or an electric bike through our partner BikeLane. The bike can be worth up to "
            "3,500 EUR including accessories. The lease lasts 36 months and the monthly fee is deducted from your "
            "gross salary, which lowers the tax you pay on it. Each employee can lease one bike at a time.",
            "Insurance against theft and a yearly service are included in the lease. Report a stolen bike to the "
            "police and to BikeLane within 48 hours, or the insurance does not pay.",
        ]),
        ("4. Parking and charging", [
            "The Barcelona office has 12 parking spaces for employees. They are assigned by lottery every 6 "
            "months, and two of them are kept for colleagues with reduced mobility. The Madrid and Lisbon offices "
            "have no parking spaces.",
            "Charging an electric car at the office chargers is free, for at most 4 hours per day, so that everyone "
            "gets a turn. Move your car when the charge is finished.",
        ]),
        ("5. Changing your choice", [
            "Join, leave or change your choice in PeopleHub with form MOB-3. Changes made before the 20th of the "
            "month apply from the next month; later changes apply a month after that.",
        ]),
        ("6. Leaving the company", [
            "The transport allowance stops in the month your contract ends. If you have a leased bike, you can "
            "either return it to BikeLane or buy it at its residual value. The remaining lease fees are not "
            "deducted from your last salary when you return the bike.",
        ]),
    ],
}


def build(spec: dict) -> Path:
    path = CORPUS / spec["file"]
    title = spec["title"]

    def decorate(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, A4[1] - 12 * mm, f"Northwind Payments | {title} | Internal use only")
        canvas.drawString(20 * mm, 10 * mm, f"Version 3.1, January 2026 | Page {document.page}")
        canvas.restoreState()

    story = [Paragraph(f"Northwind Payments {title}", STYLES["Title"]), Spacer(1, 6 * mm)]
    if spec["contents"]:
        story.append(Paragraph("Contents", STYLES["Heading2"]))
        for n, entry in enumerate(spec["contents"], start=1):
            story.append(Paragraph(f"{entry} {'.' * 60} {n + 1}", STYLES["Normal"]))
        story.append(PageBreak())
    for heading, blocks in spec["sections"]:
        story.append(Paragraph(heading, STYLES["Heading2"]))
        for block in blocks:
            if isinstance(block, tuple):
                table = Table(block[1], hAlign="LEFT")
                table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
                story += [table, Spacer(1, 3 * mm)]
            else:
                story.append(Paragraph(block, STYLES["BodyText"]))
    SimpleDocTemplate(str(path), pagesize=A4, title=f"Northwind Payments {title}", author="Northwind Payments",
                      topMargin=22 * mm, bottomMargin=20 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
                      invariant=1).build(story, onFirstPage=decorate, onLaterPages=decorate)
    return path


if __name__ == "__main__":
    for spec in (PROCUREMENT, MOBILITY):
        print(build(spec))
