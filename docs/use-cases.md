# BankMind — Banking Use Cases

This document describes five key use cases for BankMind, with example inputs, expected outputs, and notes on the underlying AI approach for each.

---

## Use Case 1: Employee Policy Q&A Assistant

### Context

A mid-sized bank may have hundreds of active internal policies, procedures, and circulars spanning payments, lending, HR, compliance, IT security, and operations. New employees spend weeks learning where documentation lives. Even experienced employees waste time hunting for the right version of a document or the exact section that covers an edge case.

### How BankMind Addresses This

The DocQAAgent indexes the entire internal document corpus. Employees query it in natural language and receive a cited, grounded answer.

### Example Interaction

**User (Operations Desk Specialist):**
```
What are the steps for processing a SWIFT MT103 payment that has been flagged 
for potential sanctions screening?
```

**BankMind:**
```
Based on the Payment Operations Manual (Section 9.4 — Sanctions-Flagged Payments, 
v3.1 — January 2024) and the AML Escalation Procedure (Section 2.1):

1. DO NOT release the payment. Place it in HOLD status in the payment gateway 
   immediately upon flag.

2. Access the Sanctions Screening System (SSS) and review the match details. 
   Note the hit type: Name Match, Address Match, or Country/Jurisdiction Match.

3. If the match is a FALSE POSITIVE (e.g., name similarity with no other 
   matching attributes), document your determination in the SSS Case Record 
   and escalate to the Sanctions Compliance team for a second-level sign-off.

4. If the match is a POTENTIAL TRUE MATCH, escalate immediately to the MLRO. 
   Do not communicate the hold to the customer until the MLRO provides guidance.

5. The MLRO must make a determination within 4 business hours (Payment Operations 
   Manual, Section 9.4.3).

6. Log all actions in the Case Management System with timestamps.

Sources:
- Payment Operations Manual v3.1, Section 9.4 (Sanctions-Flagged Payments)  
- AML Escalation Procedure v2.0, Section 2.1 (Transaction Hold Protocol)
- MLRO Contact Directory, Q3 2024
```

### Technical Notes

- Agent: `DocQAAgent`
- Technique: RAG over internal document corpus
- Key constraint: "Answer from context only" prompt instruction — ensures the answer is grounded in actual policy, not LLM-generated plausible-sounding procedures
- Source attribution is non-negotiable for employee trust and auditability

---

## Use Case 2: Loan Document Analysis & Risk Flagging

### Context

Loan officers reviewing application packages must check that all required documents are present, consistent, and within acceptable parameters. For a typical mortgage application, this involves cross-checking 15–25 documents including salary statements, property valuations, credit history, and identification documents. Missing documents or inconsistencies are common sources of delay and regulatory risk.

### How BankMind Addresses This

The ComplianceAgent can analyse an uploaded loan application package against a structured checklist of required documents and flag inconsistencies or risks.

### Example Input

```
Loan Application Package — Customer ID: C-XXXXX (anonymised for AI processing)
Loan Type: Residential Mortgage | Amount: BGN 380,000 | Term: 25 years

Documents present:
- National ID (valid, expires 2027)
- Salary certificate — employer: TechCorp Bulgaria Ltd
- Bank statements — last 6 months — BankMind Demo Bank
- Property valuation report (RICS-certified) — valuation date: 2024-09-12
- Preliminary purchase agreement — signed 2024-10-02

Documents missing:
- Credit bureau report
- Proof of own contribution (minimum 15%)
```

**User (Loan Officer):**
```
Please check this loan application package for completeness and flag any risk items.
```

**BankMind:**
```
LOAN APPLICATION RISK ASSESSMENT
Application: C-XXXXX | BGN 380,000 Mortgage | Generated: 2024-11-15

OVERALL STATUS: ⚠ INCOMPLETE — 2 blocking items require resolution

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCKING ITEMS (must resolve before approval)

[HIGH] Missing: Credit Bureau Report
Regulation: BNB Ordinance No. 9, Art. 8 — Creditworthiness Assessment
Action: Request formal credit report from CIBANK central credit register.
Processing time: typically 2–3 business days.

[HIGH] Missing: Proof of Own Contribution
Regulation: BNB Macroprudential Policy — LTV Limit (max 85% for residential)
Calculation: 380,000 BGN loan on property valued BGN 450,000 = 84.4% LTV.
At the current value, this is within limits but only marginally.
Own contribution documentation (15% = BGN 67,000) must be evidenced by 
bank statements or notarised transfer confirmation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADVISORY ITEMS (review before proceeding)

[MEDIUM] Property Valuation Age
The property valuation is dated 2024-09-12 — 64 days ago.
Per Lending Policy Section 4.2, valuations older than 90 days require 
confirmation that no material change to the property has occurred.
Monitor: if application is not completed by 2024-12-11, a new valuation 
may be required.

[LOW] Employer Verification
Salary certificate is accepted, but the Lending Procedures recommend 
employer verification call for applications above BGN 300,000.
Confirm: employer contact details verified and call logged.
```

### Technical Notes

- Agent: `ComplianceAgent` + `DocQAAgent`
- Technique: Structured checklist comparison + RAG retrieval of relevant lending regulations
- PII handling: Customer data is anonymised before being processed (Customer ID replaces name, DOB, EGN)

---

## Use Case 3: Regulatory Compliance Document Checker (BNB, ECB Directives)

### Context

Banks must constantly monitor for new regulatory requirements from BNB, ECB, and EU legislative bodies, and assess how they affect internal policies and procedures. Compliance officers spend days reading regulatory texts and cross-referencing them against existing internal policies. Newly published directives or circulars often require rapid impact assessment.

### How BankMind Addresses This

The ComplianceAgent checks internal documents and communications against the regulatory rule corpus and flags specific gaps or required updates.

### Example Interaction

**User (Compliance Officer):**
```
We've just updated our AML Monitoring Policy. Can you check whether it covers 
all the requirements from the 6th EU AML Directive and BNB Ordinance 38?

[Document text pasted below]

"3.1 Transaction Monitoring. The Bank monitors all transactions in real-time 
using automated rules-based screening. Transactions above EUR 10,000 equivalent 
are subject to enhanced review. Suspicious patterns including structuring, 
unusual cross-border activity, and transactions inconsistent with customer 
profile trigger a case for analyst review. All STRs are filed with the relevant 
authorities within 5 business days."
```

**BankMind:**
```
COMPLIANCE ASSESSMENT — AML Monitoring Policy Section 3.1
Framework: 6th EU AML Directive + BNB Ordinance No. 38
Assessed: 2024-11-15 14:32 | DRAFT — Requires Compliance Officer Review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL RISK LEVEL: MEDIUM

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINDING 1: STR Filing Timeline — Possible Non-Compliance
Risk Level: HIGH
Regulation: BNB Ordinance No. 38, Article 18
Issue: The policy states STRs are filed "within 5 business days." BNB Ordinance 
No. 38, Art. 18 requires filing with SANS within 3 working days.
Remediation: Update Section 3.1 to read "within 3 working days" in line 
with BNB Ordinance No. 38 Art. 18. Review any recent STRs filed between 
day 3 and day 5 to assess whether retrospective notification to SANS is required.

FINDING 2: Transaction Threshold — BNB vs EU
Risk Level: MEDIUM  
Regulation: BNB Ordinance No. 38, Article 10
Issue: The policy monitors transactions above "EUR 10,000 equivalent." 
BNB Ordinance No. 38, Art. 10 specifies the threshold in BGN (BGN 30,000), 
not EUR. Due to exchange rate fluctuation, EUR 10,000 ≈ BGN 19,550 — 
significantly below the BGN 30,000 threshold. 
Note: Your policy is more conservative (catches more transactions), which 
is compliant, but the discrepancy in currency should be documented to avoid 
confusion in regulatory examinations.
Recommendation: Add a footnote referencing both thresholds and the 
EUR/BGN equivalence basis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AREAS THAT APPEAR COMPLIANT

- Real-time monitoring system: consistent with 6th AMLD transaction monitoring requirements
- Enhanced review for high-value transactions: consistent with ECDD requirements
- Structuring detection mentioned: consistent with 6th AMLD Art. 3 typologies
- Cross-border activity monitoring: consistent with 5th AMLD high-risk jurisdiction requirements
```

### Technical Notes

- Agent: `ComplianceAgent`
- LLM model: GPT-4o (full model, not mini) — regulatory nuance requires strongest reasoning
- Temperature: 0 — compliance assessments must be deterministic and reproducible

---

## Use Case 4: Monthly Report Auto-Drafting

### Context

Operations managers and risk managers spend significant time every month producing reports for senior management and the board. These reports follow predictable structures but require translating raw operational data into coherent narrative commentary. The first draft of such a report can take half a day to produce.

### How BankMind Addresses This

The ReportGenerationAgent accepts structured data inputs and generates a complete, professional first draft that the manager reviews, adjusts, and approves — reducing the report production cycle from half a day to 30 minutes.

### Example Input

**User (Operations Manager):**
```
Generate the monthly operational report for November 2024 using this data:
- Total transactions processed: 127,543
- SEPA outbound: 48,201 | SEPA inbound: 53,102
- BISERA domestic: 26,240
- Failed/returned: 287 (0.22% return rate)
- Exceptions raised: 22 (P1: 1, P2: 4, P3/P4: 17)
- SLA resolution rate: 96%
- Compliance notes: DORA readiness assessment completed; no critical gaps
- Outstanding: Q4 capital reporting due Dec 31
```

**BankMind:**
```
MONTHLY OPERATIONAL REPORT — NOVEMBER 2024
Prepared by: BankMind AI Drafting Assistant
Date Generated: 15 November 2024, 14:45
Status: DRAFT — Requires Management Review and Approval

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EXECUTIVE SUMMARY
November 2024 marked a stable operational month, with 127,543 transactions 
processed at a return rate of 0.22% — below the 0.30% internal threshold. 
One critical (P1) exception was raised and resolved within SLA. The DORA 
readiness assessment was completed with no critical gaps identified, 
representing a significant compliance milestone for Q4.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. PAYMENT OPERATIONS
   Total Transactions Processed:  127,543
   SEPA Credit Transfers (out):    48,201
   SEPA Credit Transfers (in):     53,102
   Domestic Transfers (BISERA):    26,240
   Failed / Returned:                 287
   Return Rate:                     0.22%

Transaction volumes remained consistent with October 2024 levels. The 0.22% 
return rate continues to track below the 0.30% internal operational threshold 
and well below the industry average of approximately 0.40% for comparable 
institutions. SEPA inbound volumes slightly exceeded outbound, consistent with 
seasonal pre-holiday payment patterns observed in prior years.

[continues...]
```

### Technical Notes

- Agent: `ReportGenerationAgent`
- Key design: Numbers are bound into the template before the LLM generates commentary — LLM cannot hallucinate figures
- Requires human review: All drafts are clearly marked as DRAFT

---

## Use Case 5: Customer Complaint Summarisation & Routing

### Context

Customer service centres receive hundreds of complaints per day through multiple channels (email, call centre transcripts, branch forms, online portal). Each complaint must be:
- Summarised (for the case file)
- Categorised by type (payments, lending, fees, service quality, etc.)
- Prioritised (urgent regulatory complaints vs. routine service issues)
- Routed to the correct handling team

This triage process is time-consuming and inconsistent when done manually.

### How BankMind Addresses This

The DocQAAgent (with a complaint-specific prompt) summarises each complaint, classifies it, assigns a priority, and suggests the appropriate handling team — producing a structured triage record that a customer service supervisor reviews and confirms.

### Example Input

```
Original complaint (received via email, 2024-11-14):
---
Subject: OUTRAGEOUS fee charged without warning

I am writing to complain in the strongest possible terms. Last month I noticed 
a charge of BGN 47.00 on my account that I was never told about. When I called 
your contact centre I was put on hold for 35 minutes and then disconnected. 
When I finally got through, the agent told me it was an "account maintenance 
fee" but could not tell me when this was introduced or where it is documented.

I am a customer of 12 years. This is not acceptable. I want a full refund and 
a written explanation of when this fee was introduced, where it is documented, 
and why I was not informed in advance. If I do not receive a satisfactory response 
within 10 days I will file a formal complaint with the Financial Supervision Commission.
---
```

**BankMind:**
```
COMPLAINT TRIAGE RECORD
Generated: 2024-11-14 | Reference: AUTO-2024-11-14-0034 | Status: DRAFT — Requires Review

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY (2 sentences)
The customer disputes an undisclosed BGN 47.00 account maintenance fee and 
received inadequate customer service when seeking clarification. They demand 
a refund and formal written explanation, and have threatened escalation to 
the Financial Supervision Commission (FSC) within 10 days.

CATEGORY:  Fees & Charges — Undisclosed / Unexpected Fee

PRIORITY:  HIGH
Reason for high priority: Explicit mention of FSC escalation establishes a 
10-day regulatory response clock. FSC complaints require formal written 
responses within prescribed timelines.

HANDLING TEAM:  Retail Banking Customer Relations
Escalation required: Yes — Flag to Compliance team for FSC escalation risk tracking

REGULATORY NOTES:
BNB Consumer Protection Guidelines require advance notification of fee changes. 
If the account maintenance fee was introduced without 30-day advance notice, 
there may be a grounds for refund and a potential regulatory reporting obligation.
Recommend: Compliance team to confirm whether proper notification was issued.

REQUIRED ACTIONS:
1. Acknowledge complaint in writing within 2 business days (regulatory requirement)
2. Pull account history and fee notice records for this customer
3. Confirm whether advance notice of the fee change was issued and documented
4. Compliance review: assess FSC escalation risk
5. Prepare formal response within 10 calendar days
```

### Technical Notes

- Agent: `DocQAAgent` (with complaint-specific system prompt)
- Technique: Classification + summarisation + regulatory context lookup
- Key output fields: category, priority, handling team, regulatory notes — these are used for downstream workflow routing
- Privacy: Customer name, account number, and contact details are stripped before AI processing; only complaint text is sent to the LLM
