"""
ComplianceAgent — Regulatory Compliance Checking Agent

Banking operations in Bulgaria are governed by a layered regulatory framework:
- Bulgarian National Bank (BNB) — primary prudential supervisor, issues Ordinances
  and Regulations (e.g., Ordinance No. 7, No. 8, No. 38)
- European Central Bank (ECB) — Single Supervisory Mechanism (SSM) for significant
  institutions; issues Regulations and Guidelines
- EU Directives — CRD/CRR (capital requirements), PSD2 (payments), AML Directive,
  DORA (Digital Operational Resilience Act)
- GDPR (Regulation 2016/679) — data protection, critical for customer data handling

This agent checks documents and communications against a rules corpus derived
from these regulatory sources. It:
1. Searches the regulatory document vector store for relevant rules
2. Compares the input document against those rules using an LLM as a reasoning engine
3. Returns a structured assessment with flagged items and remediation suggestions

Design note on false negatives vs. false positives:
----------------------------------------------------
In a compliance context, it is far better to produce a false positive (flag
something that turns out to be compliant) than a false negative (miss a genuine
compliance issue). The prompts are therefore calibrated to err on the side of
caution — the output should always be reviewed by a compliance professional.
"""

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("bankmind.agents.compliance")

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. ComplianceAgent will run in mock mode.")


# ---------------------------------------------------------------------------
# Regulatory rules reference — a lightweight in-memory rule set
# In production this would be a separate vector store index of regulatory docs
# ---------------------------------------------------------------------------

# Key BNB and EU regulatory requirements distilled into checkable items.
# This is used as the grounding context when no external regulatory document
# corpus is indexed. A production system would replace this with a full
# RAG lookup against the actual regulatory text.
REGULATORY_RULES = {
    "BNB": [
        {
            "id": "BNB-ORD7-ART15",
            "framework": "BNB Ordinance No. 7",
            "requirement": "Internal control systems must include documented procedures for risk identification and mitigation.",
            "risk_area": "Internal Controls",
        },
        {
            "id": "BNB-ORD8-ART22",
            "framework": "BNB Ordinance No. 8",
            "requirement": "Capital adequacy reports must be submitted quarterly and include Tier 1 and Tier 2 capital breakdown.",
            "risk_area": "Capital Adequacy",
        },
        {
            "id": "BNB-ORD38-ART10",
            "framework": "BNB Ordinance No. 38",
            "requirement": "Banks must maintain records of all transactions above BGN 30,000 for AML monitoring purposes.",
            "risk_area": "AML / CTF",
        },
        {
            "id": "BNB-ORD38-ART18",
            "framework": "BNB Ordinance No. 38",
            "requirement": "Suspicious transaction reports (STRs) must be filed with SANS (Financial Intelligence Agency) within 3 working days.",
            "risk_area": "AML / CTF",
        },
        {
            "id": "BNB-CONSUMER-ART5",
            "framework": "BNB Consumer Protection Guidelines",
            "requirement": "Customer communications must include clear disclosure of fees, interest rates, and cooling-off periods.",
            "risk_area": "Consumer Protection",
        },
    ],
    "ECB": [
        {
            "id": "ECB-SSM-ICAAP",
            "framework": "ECB ICAAP Guide",
            "requirement": "ICAAP documentation must demonstrate comprehensive identification of material risks.",
            "risk_area": "Risk Management",
        },
        {
            "id": "ECB-CRR2-ART92",
            "framework": "CRR2 Article 92",
            "requirement": "Total capital ratio must be maintained at a minimum of 8% of total risk-weighted exposure.",
            "risk_area": "Capital Adequacy",
        },
    ],
    "GDPR": [
        {
            "id": "GDPR-ART13",
            "framework": "GDPR Article 13",
            "requirement": "Data subjects must be informed of the purposes and legal basis for processing their personal data at the time of collection.",
            "risk_area": "Data Protection",
        },
        {
            "id": "GDPR-ART17",
            "framework": "GDPR Article 17",
            "requirement": "Data subjects have the right to erasure. Procedures must exist to process such requests within 30 days.",
            "risk_area": "Data Protection",
        },
        {
            "id": "GDPR-ART32",
            "framework": "GDPR Article 32",
            "requirement": "Appropriate technical and organisational measures must be implemented to ensure data security.",
            "risk_area": "Data Protection",
        },
    ],
    "AML": [
        {
            "id": "AML6-ART3",
            "framework": "6th EU AML Directive",
            "requirement": "Criminal liability for money laundering extends to all persons who assist or facilitate ML activities.",
            "risk_area": "AML / CTF",
        },
        {
            "id": "AML5-CDD",
            "framework": "5th EU AML Directive",
            "requirement": "Enhanced Customer Due Diligence (ECDD) must be applied for high-risk customers, PEPs, and transactions from high-risk jurisdictions.",
            "risk_area": "AML / CTF",
        },
    ],
}

COMPLIANCE_SYSTEM_PROMPT = """You are a regulatory compliance specialist at a Bulgarian commercial bank.
You have deep expertise in BNB regulations, ECB guidelines, EU Directives (CRD, PSD2, AML, GDPR), and DORA.
Your role is to assess documents for compliance gaps and risks.

When checking a document:
- Be thorough and conservative — flag potential issues even if they are ambiguous
- Reference specific regulatory requirements by name and article number where possible
- Rate each finding as: HIGH RISK, MEDIUM RISK, or LOW RISK / ADVISORY
- Suggest specific remediation actions for each finding
- Never fabricate regulatory references — only cite frameworks from the provided context

Format your response as structured findings, not prose paragraphs.
"""


class ComplianceAgent:
    """
    Regulatory Compliance Checking Agent.

    This agent evaluates banking documents and communications against a
    curated set of regulatory requirements from BNB, ECB, EU Directives,
    and GDPR. It is designed as a first-pass screening tool — its output
    should always be reviewed by a qualified compliance professional.

    The agent does NOT:
    - Provide legal advice
    - Make final compliance determinations
    - Replace the bank's formal compliance review process

    The agent DOES:
    - Accelerate initial document review by flagging potential issues
    - Provide regulatory context for flagged items
    - Suggest preliminary remediation directions

    Attributes:
        settings: Application settings.
        llm: LLM instance for compliance analysis.
    """

    def __init__(self, settings=None):
        """
        Initialise the ComplianceAgent.

        Args:
            settings: Application settings instance.
        """
        self.settings = settings
        self._mock_mode = not LANGCHAIN_AVAILABLE

        if not self._mock_mode:
            self._llm = self._build_llm()
        else:
            self._llm = None

        logger.info("ComplianceAgent initialised. Mock mode: %s", self._mock_mode)

    def _build_llm(self):
        """
        Builds the LLM for compliance analysis.

        GPT-4o is used (not mini) because compliance checking requires
        careful multi-step reasoning over regulatory text — this is
        exactly the type of task where the larger model's improved
        instruction following and reasoning significantly outperforms
        the smaller model. The cost differential is justified given the
        regulatory stakes of missing a compliance issue.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = getattr(self.settings, "llm_model_compliance", "gpt-4o") if self.settings else "gpt-4o"

        return ChatOpenAI(
            model=model,
            temperature=0,      # Zero temperature for deterministic compliance assessments
            api_key=api_key or "placeholder",
            max_tokens=3000,
        )

    def check_document(
        self,
        document_text: str,
        regulation_context: str = "BNB",
    ) -> dict:
        """
        Check a document for compliance with the specified regulatory framework.

        This is the primary method of the ComplianceAgent. It:
        1. Retrieves relevant rules for the specified regulatory context
        2. Constructs a prompt that provides both the document and the rules
        3. Asks the LLM to identify gaps and issues
        4. Returns a structured compliance report

        Args:
            document_text: The text of the document to check.
            regulation_context: Which regulatory framework to apply.
                                 Options: 'BNB', 'ECB', 'GDPR', 'AML', 'ALL'.

        Returns:
            A dictionary containing:
            - 'summary': High-level compliance assessment
            - 'findings': List of specific compliance issues found
            - 'risk_level': Overall risk level ('HIGH', 'MEDIUM', 'LOW', 'CLEAN')
            - 'references': List of regulatory references cited
            - 'checked_at': ISO timestamp of the check
        """
        if self._mock_mode or self._llm is None:
            return self._mock_check_document(document_text, regulation_context)

        # Collect applicable rules
        if regulation_context == "ALL":
            all_rules = []
            for rules in REGULATORY_RULES.values():
                all_rules.extend(rules)
        else:
            all_rules = REGULATORY_RULES.get(regulation_context, REGULATORY_RULES["BNB"])

        rules_context = "\n".join(
            f"[{r['id']}] {r['framework']}: {r['requirement']} (Risk area: {r['risk_area']})"
            for r in all_rules
        )

        user_prompt = f"""
Review the following document for compliance with the {regulation_context} regulatory requirements listed below.

=== REGULATORY REQUIREMENTS ===
{rules_context}

=== DOCUMENT TO REVIEW ===
{document_text[:5000]}  

=== INSTRUCTIONS ===
For each compliance issue found, provide:
1. FINDING: What is the issue
2. REGULATION: Which specific rule it relates to (use the rule ID above)
3. RISK LEVEL: HIGH / MEDIUM / LOW
4. REMEDIATION: Specific action to address the issue

Start your response with a one-sentence overall compliance summary.
Then list all findings. If the document appears fully compliant, state that explicitly.
"""

        try:
            messages = [
                SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = self._llm.invoke(messages)
            assessment_text = response.content

            # Determine overall risk level from the response
            risk_level = self._extract_risk_level(assessment_text)

            return {
                "summary": assessment_text,
                "findings": self._parse_findings(assessment_text),
                "risk_level": risk_level,
                "references": [r["id"] for r in all_rules],
                "regulation_context": regulation_context,
                "checked_at": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("Compliance check failed: %s", exc, exc_info=True)
            return {
                "summary": "Compliance check failed due to a system error.",
                "findings": [],
                "risk_level": "UNKNOWN",
                "references": [],
                "checked_at": datetime.now().isoformat(),
                "error": str(exc),
            }

    def flag_risks(self, document_text: str) -> list[dict]:
        """
        Extract and return a structured list of risk flags from a document.

        This is a convenience method that calls check_document and returns
        only the findings list, suitable for use in automated screening
        pipelines (e.g., loan document ingestion workflows).

        Args:
            document_text: The document text to screen.

        Returns:
            A list of risk flag dictionaries, each with 'description',
            'risk_level', and 'regulation' keys.
        """
        result = self.check_document(document_text, regulation_context="ALL")
        return result.get("findings", [])

    def suggest_remediation(self, finding: dict) -> str:
        """
        Generate a detailed remediation plan for a specific compliance finding.

        This method is called when a compliance reviewer wants to drill down
        into a specific finding and get more detailed guidance on how to
        address it.

        Args:
            finding: A finding dict as returned by check_document() or flag_risks().
                     Expected keys: 'description', 'risk_level', 'regulation'.

        Returns:
            A string with specific, actionable remediation steps.
        """
        if self._mock_mode or self._llm is None:
            return (
                "[DEMO MODE] Remediation steps would be generated here based on the "
                f"finding: {finding.get('description', 'Unknown finding')}"
            )

        user_prompt = f"""
A compliance review has identified the following finding in a banking document:

Finding: {finding.get('description', 'Not specified')}
Risk Level: {finding.get('risk_level', 'Not specified')}
Relevant Regulation: {finding.get('regulation', 'Not specified')}

Please provide:
1. A clear explanation of why this is a compliance concern
2. Specific, actionable steps to remediate this finding
3. Any documentation or evidence that should be gathered
4. An estimated timeline for remediation
5. Who should own this remediation (role/function, not individual name)

Be specific and practical. This guidance will be used by the bank's compliance team.
"""

        try:
            messages = [
                SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = self._llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error("Remediation suggestion failed: %s", exc)
            return "Could not generate remediation guidance. Please consult the compliance team directly."

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _extract_risk_level(self, assessment_text: str) -> str:
        """
        Extracts the highest risk level mentioned in an assessment.

        This is a simple heuristic — a production system might parse a
        structured JSON response from the LLM instead.
        """
        text_upper = assessment_text.upper()
        if "HIGH RISK" in text_upper:
            return "HIGH"
        elif "MEDIUM RISK" in text_upper:
            return "MEDIUM"
        elif "LOW RISK" in text_upper:
            return "LOW"
        elif "COMPLIANT" in text_upper or "NO ISSUES" in text_upper:
            return "CLEAN"
        return "UNKNOWN"

    def _parse_findings(self, assessment_text: str) -> list[dict]:
        """
        Parses the LLM's free-text assessment into structured findings.

        This is a lightweight parser — in production it would be more robust,
        or we would instruct the LLM to return JSON directly.
        """
        findings = []
        lines = assessment_text.split("\n")
        current_finding = {}

        for line in lines:
            line = line.strip()
            if line.startswith("FINDING:") or line.startswith("1. FINDING:"):
                if current_finding:
                    findings.append(current_finding)
                current_finding = {"description": line.replace("FINDING:", "").replace("1. FINDING:", "").strip()}
            elif line.startswith("RISK LEVEL:"):
                current_finding["risk_level"] = line.replace("RISK LEVEL:", "").strip()
            elif line.startswith("REGULATION:"):
                current_finding["regulation"] = line.replace("REGULATION:", "").strip()
            elif line.startswith("REMEDIATION:"):
                current_finding["remediation"] = line.replace("REMEDIATION:", "").strip()

        if current_finding:
            findings.append(current_finding)

        return findings

    def _mock_check_document(self, document_text: str, regulation_context: str) -> dict:
        """
        Returns a mock compliance check result for testing and demo purposes.
        """
        return {
            "summary": (
                f"[DEMO MODE] This document has been assessed against {regulation_context} regulations. "
                f"In a live environment, the AI would perform a detailed compliance check against "
                f"all relevant {regulation_context} requirements and flag any gaps or risks."
            ),
            "findings": [
                {
                    "description": "Sample finding: Customer consent language may not meet the specificity requirements",
                    "risk_level": "MEDIUM",
                    "regulation": "GDPR-ART13",
                    "remediation": "Review consent language against GDPR Article 13 requirements",
                },
                {
                    "description": "Sample finding: Transaction monitoring thresholds not explicitly stated",
                    "risk_level": "LOW",
                    "regulation": "BNB-ORD38-ART10",
                    "remediation": "Add explicit reference to BGN 30,000 threshold per BNB Ordinance No. 38",
                },
            ],
            "risk_level": "MEDIUM",
            "references": [r["id"] for r in REGULATORY_RULES.get(regulation_context, [])],
            "regulation_context": regulation_context,
            "checked_at": datetime.now().isoformat(),
            "mock": True,
        }
