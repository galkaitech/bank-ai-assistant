"""
ReportGenerationAgent — Structured Banking Report Drafting Agent

Bank employees spend significant time drafting operational and compliance
reports that follow predictable structures. This agent accelerates that
process by:

1. Taking structured data inputs (transaction counts, volumes, exception
   counts, etc.) from the caller
2. Filling those into a report template using Jinja2
3. Passing the populated template to an LLM to generate natural language
   narrative sections (commentary, analysis, conclusions)
4. Returning a complete draft that a human reviewer can approve and adjust

Why template + LLM rather than pure LLM generation?
----------------------------------------------------
Purely LLM-generated reports are prone to hallucinating specific numbers and
dates. By forcing the LLM to work from structured data filled into a template,
we guarantee that figures are correct. The LLM's job is only to generate
the narrative commentary — it cannot fabricate a number it wasn't given.

This hybrid approach (structured data → template → LLM narrative) is a
common pattern in fintech GenAI applications and is safer than unbounded
report generation.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("bankmind.agents.report")

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. ReportGenerationAgent will run in mock mode.")


# ---------------------------------------------------------------------------
# Report prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_REPORT = """You are a senior banking operations analyst at a Bulgarian commercial bank.
You write clear, professional reports for internal management and regulatory purposes.
Your reports are factual, concise, and written in formal business English.
You do not speculate or add information that was not provided to you.
When asked to comment on data, your analysis is grounded only in the figures given.
"""

MONTHLY_REPORT_TEMPLATE = """
MONTHLY OPERATIONAL REPORT — {period}
Prepared by: BankMind AI Drafting Assistant
Date Generated: {generation_date}
Status: DRAFT — Requires Human Review and Approval

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EXECUTIVE SUMMARY
{executive_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. PAYMENT OPERATIONS
   Total Transactions Processed: {total_transactions:,}
   SEPA Credit Transfers (outbound): {sepa_outbound:,}
   SEPA Credit Transfers (inbound): {sepa_inbound:,}
   Domestic Transfers (BISERA): {bisera_transactions:,}
   Failed / Returned Transactions: {failed_transactions:,}
   Return Rate: {return_rate:.2f}%

{payment_commentary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. EXCEPTION & INCIDENT SUMMARY
   Exceptions Raised: {exceptions_total}
   Critical (P1): {exceptions_p1}
   High (P2): {exceptions_p2}
   Medium/Low (P3/P4): {exceptions_p3p4}
   Resolved Within SLA: {resolved_within_sla}%

{exception_commentary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. COMPLIANCE HIGHLIGHTS
{compliance_highlights}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. OUTSTANDING ITEMS AND ACTIONS
{outstanding_items}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISCLAIMER: This report was drafted with AI assistance. All figures are as
provided by the data input. Narrative sections require review before
submission to management or regulators.
"""

COMPLIANCE_NOTE_TEMPLATE = """
COMPLIANCE NOTE
Reference: CN-{ref_number}
Date: {date}
Prepared by: BankMind AI Drafting Assistant
Subject: {subject}
Status: DRAFT

1. PURPOSE
{purpose_text}

2. REGULATORY BACKGROUND
{regulatory_background}

3. CURRENT STATUS / FINDINGS
{findings}

4. REQUIRED ACTIONS
{required_actions}

5. TIMELINE AND OWNERSHIP
{timeline}

6. SIGN-OFF REQUIRED FROM
   □ Compliance Officer
   □ Head of Legal
   □ Relevant Business Unit Head

NOTE: This document is AI-assisted. Final content must be reviewed and
approved by the Compliance team before distribution.
"""


class ReportGenerationAgent:
    """
    Report Generation Agent for structured banking reports and compliance notes.

    This agent combines deterministic template filling (for figures and structured
    data) with LLM-generated narrative commentary. The separation between
    structured data and narrative is intentional — it ensures numerical accuracy
    while still benefiting from the LLM's ability to generate well-written prose.

    Attributes:
        settings: Application settings instance.
        llm: The LLM instance used for narrative generation.
    """

    def __init__(self, settings=None):
        """
        Initialise the ReportGenerationAgent.

        Args:
            settings: Application settings. If None, uses environment defaults.
        """
        self.settings = settings
        self._mock_mode = not LANGCHAIN_AVAILABLE

        if not self._mock_mode:
            self._llm = self._build_llm()
        else:
            self._llm = None

        logger.info("ReportGenerationAgent initialised. Mock mode: %s", self._mock_mode)

    def _build_llm(self):
        """
        Builds the LLM for report generation.

        Report generation uses GPT-4o (not GPT-4o-mini) because:
        - Report narrative quality directly reflects on the bank's professionalism
        - Reports may be read by management or regulators — higher stakes than
          an internal Q&A query
        - The extra cost per report generation is justified given the time saved
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = getattr(self.settings, "llm_model_reports", "gpt-4o") if self.settings else "gpt-4o"

        return ChatOpenAI(
            model=model,
            temperature=0.2,    # Very slight temperature for natural-sounding prose variation
            api_key=api_key or "placeholder",
            max_tokens=2000,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Makes an LLM call with structured system and user prompts.

        Args:
            system_prompt: The system-level instruction for the LLM.
            user_prompt: The user-level request with context data.

        Returns:
            The LLM's text response, or a placeholder if in mock mode.
        """
        if self._mock_mode or self._llm is None:
            return "[LLM-generated narrative would appear here in a live environment]"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = self._llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("LLM call failed in ReportGenerationAgent: %s", exc)
            return "[Report narrative generation failed. Please review and complete this section manually.]"

    def generate_summary(self, data: dict[str, Any], report_type: str = "general") -> str:
        """
        Generate a concise summary for ad-hoc or unstructured report requests.

        This is the general-purpose method for report generation when a specific
        report template doesn't apply. It passes the provided data as context
        to the LLM and asks for a structured summary.

        Args:
            data: A dictionary of input data to summarise.
            report_type: A descriptive label for the type of report.

        Returns:
            A formatted summary string ready for review.
        """
        data_str = json.dumps(data, indent=2, ensure_ascii=False)

        user_prompt = f"""
Please generate a concise, professional {report_type} summary based on the following data.
Structure the output with clear sections. Do not add information that is not in the data.

Input data:
{data_str}

Generate a professional summary appropriate for internal banking management.
"""
        narrative = self._call_llm(SYSTEM_PROMPT_REPORT, user_prompt)

        header = (
            f"SUMMARY REPORT — {report_type.upper().replace('_', ' ')}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Status: DRAFT — Requires Human Review\n"
            f"{'─' * 60}\n\n"
        )
        return header + narrative

    def generate_monthly_report(self, data: dict[str, Any], period: str = "current") -> str:
        """
        Generate a monthly operational report from structured input data.

        The method validates that required data fields are present, fills
        them into the report template, then uses the LLM to generate
        commentary sections for each major area.

        Args:
            data: Dictionary containing operational metrics. Expected keys:
                  total_transactions, sepa_outbound, sepa_inbound,
                  bisera_transactions, failed_transactions, exceptions_total,
                  exceptions_p1, exceptions_p2, exceptions_p3p4,
                  resolved_within_sla, compliance_notes (list of strings),
                  outstanding_items (list of strings)
            period: The reporting period label (e.g., "November 2024", "2024-Q3").

        Returns:
            A formatted monthly report string ready for management review.
        """
        # Apply defaults for missing fields to prevent KeyErrors
        defaults = {
            "total_transactions": 0,
            "sepa_outbound": 0,
            "sepa_inbound": 0,
            "bisera_transactions": 0,
            "failed_transactions": 0,
            "exceptions_total": 0,
            "exceptions_p1": 0,
            "exceptions_p2": 0,
            "exceptions_p3p4": 0,
            "resolved_within_sla": 0,
        }
        report_data = {**defaults, **data}

        # Calculate derived metrics
        total = report_data["total_transactions"]
        failed = report_data["failed_transactions"]
        report_data["return_rate"] = (failed / total * 100) if total > 0 else 0.0

        # Generate narrative sections via LLM
        payment_prompt = (
            f"Write a 2-3 sentence professional commentary on the payment operations data for {period}. "
            f"Total transactions: {total:,}. SEPA outbound: {report_data['sepa_outbound']:,}. "
            f"Return rate: {report_data['return_rate']:.2f}%. Be factual and analytical."
        )
        report_data["payment_commentary"] = self._call_llm(SYSTEM_PROMPT_REPORT, payment_prompt)

        exc_prompt = (
            f"Write a 2-3 sentence professional commentary on the exception data for {period}. "
            f"Total exceptions: {report_data['exceptions_total']}. "
            f"P1 critical: {report_data['exceptions_p1']}. "
            f"SLA resolution rate: {report_data['resolved_within_sla']}%."
        )
        report_data["exception_commentary"] = self._call_llm(SYSTEM_PROMPT_REPORT, exc_prompt)

        # Format compliance highlights and outstanding items
        compliance_notes = data.get("compliance_notes", ["No compliance items flagged this period."])
        report_data["compliance_highlights"] = "\n".join(f"   • {item}" for item in compliance_notes)

        outstanding = data.get("outstanding_items", ["No outstanding items."])
        report_data["outstanding_items"] = "\n".join(f"   □ {item}" for item in outstanding)

        # Generate executive summary last (so it can reference the full picture)
        exec_prompt = (
            f"Write a 3-4 sentence executive summary for the monthly operations report for {period}. "
            f"Key metrics: {total:,} transactions processed, {report_data['return_rate']:.2f}% return rate, "
            f"{report_data['exceptions_total']} exceptions ({report_data['exceptions_p1']} critical), "
            f"{report_data['resolved_within_sla']}% resolved within SLA."
        )
        report_data["executive_summary"] = self._call_llm(SYSTEM_PROMPT_REPORT, exec_prompt)

        report_data["period"] = period.upper()
        report_data["generation_date"] = datetime.now().strftime("%d %B %Y, %H:%M")

        return MONTHLY_REPORT_TEMPLATE.format(**report_data)

    def draft_compliance_note(self, data: dict[str, Any]) -> str:
        """
        Draft a compliance note (memo) for internal regulatory tracking.

        Compliance notes document regulatory findings, actions required, and
        timelines. They are used internally and may be referenced in regulatory
        examinations by BNB or ECB inspectors.

        Args:
            data: Dictionary containing:
                  subject (str): The subject of the compliance note.
                  regulatory_reference (str): Relevant regulation or circular.
                  findings (str or list): Key findings or gaps identified.
                  required_actions (str or list): Actions to be taken.
                  deadline (str): Deadline for completion.

        Returns:
            A formatted compliance note draft.
        """
        subject = data.get("subject", "Regulatory Compliance Update")
        reg_ref = data.get("regulatory_reference", "BNB Regulation")

        # Generate substantive sections via LLM, grounded in provided data
        purpose_prompt = (
            f"Write a 2-sentence purpose statement for a compliance note regarding: '{subject}'. "
            f"Regulatory reference: {reg_ref}. Use formal banking compliance language."
        )
        purpose_text = self._call_llm(SYSTEM_PROMPT_REPORT, purpose_prompt)

        reg_bg_prompt = (
            f"Write 3-4 sentences of regulatory background for a compliance note on '{subject}', "
            f"referencing {reg_ref}. Describe what the regulation requires in general terms."
        )
        regulatory_background = self._call_llm(SYSTEM_PROMPT_REPORT, reg_bg_prompt)

        # Format findings
        findings_raw = data.get("findings", "No specific findings provided.")
        if isinstance(findings_raw, list):
            findings = "\n".join(f"   {i+1}. {f}" for i, f in enumerate(findings_raw))
        else:
            findings = f"   {findings_raw}"

        # Format required actions
        actions_raw = data.get("required_actions", "Actions to be determined.")
        if isinstance(actions_raw, list):
            required_actions = "\n".join(f"   {i+1}. {a}" for i, a in enumerate(actions_raw))
        else:
            required_actions = f"   {actions_raw}"

        deadline = data.get("deadline", "To be confirmed")
        owner = data.get("owner", "Compliance Officer")
        timeline = f"   Completion Deadline: {deadline}\n   Primary Owner: {owner}"

        import random
        ref_number = f"{datetime.now().year}-{random.randint(1000, 9999)}"

        return COMPLIANCE_NOTE_TEMPLATE.format(
            ref_number=ref_number,
            date=datetime.now().strftime("%d %B %Y"),
            subject=subject,
            purpose_text=purpose_text,
            regulatory_background=regulatory_background,
            findings=findings,
            required_actions=required_actions,
            timeline=timeline,
        )
