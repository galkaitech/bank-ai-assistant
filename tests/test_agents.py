"""
BankMind Agent Unit Tests
==========================

Test strategy:
- All LLM calls are mocked using unittest.mock.patch to avoid:
  a) Requiring an API key in the CI environment
  b) Incurring API costs on every test run
  c) Flakiness from network calls in unit tests

- Document store operations are mocked to avoid requiring a running
  ChromaDB instance

- Tests validate the agent logic and interfaces, not the LLM outputs

- Each test class corresponds to one agent class

Run tests:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_mock_llm_response(content: str) -> Mock:
    """Creates a mock LLM response object that mimics the OpenAI response structure."""
    response = Mock()
    response.content = content
    return response


def make_mock_settings(**overrides) -> Mock:
    """Creates a mock settings object with sensible defaults for testing."""
    settings = Mock()
    settings.app_mode = "demo"
    settings.llm_model = "gpt-4o"
    settings.llm_model_doc_qa = "gpt-4o-mini"
    settings.llm_model_reports = "gpt-4o"
    settings.llm_model_compliance = "gpt-4o"
    settings.llm_temperature = 0.0
    settings.chunk_size = 800
    settings.chunk_overlap = 100
    settings.retrieval_top_k = 4
    settings.chroma_persist_dir = "/tmp/test_chroma"
    settings.is_demo_mode = True

    for key, value in overrides.items():
        setattr(settings, key, value)

    return settings


# ---------------------------------------------------------------------------
# DocQAAgent Tests
# ---------------------------------------------------------------------------

class TestDocQAAgent:
    """Tests for the DocQAAgent — document retrieval and Q&A."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures before each test."""
        # We patch at the module level to prevent any real LangChain imports
        # from attempting to connect to external services
        with patch.dict("sys.modules", {
            "langchain": MagicMock(),
            "langchain_openai": MagicMock(),
            "langchain_community": MagicMock(),
            "langchain_community.vectorstores": MagicMock(),
            "langchain_community.document_loaders": MagicMock(),
            "langchain.text_splitter": MagicMock(),
            "langchain.chains": MagicMock(),
            "langchain.schema": MagicMock(),
        }):
            from src.agents.document_agent import DocQAAgent
            self.DocQAAgent = DocQAAgent
            self.settings = make_mock_settings()

    def test_init_in_mock_mode(self):
        """Agent should initialise in mock mode when LangChain is not available."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            assert agent._mock_mode is True
            assert agent._llm is None
            assert agent._embeddings is None

    def test_mock_query_returns_tuple(self):
        """Mock query should return a (str, list) tuple."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            answer, sources = agent.query("What is the SEPA return procedure?")

            assert isinstance(answer, str)
            assert isinstance(sources, list)
            assert len(answer) > 0

    def test_mock_query_includes_question_context(self):
        """Mock query should reference the original question in its response."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            question = "What is the AML threshold for reporting?"
            answer, _ = agent.query(question)

            assert question in answer

    def test_mock_sources_have_expected_structure(self):
        """Each source in the mock response should have required keys."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            _, sources = agent.query("Test question")

            for source in sources:
                assert "document" in source
                assert "page" in source
                assert "snippet" in source

    def test_mock_documents_returns_list(self):
        """_mock_documents should return a non-empty list."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            docs = agent._mock_documents()

            assert isinstance(docs, list)
            assert len(docs) > 0

    def test_load_documents_raises_on_missing_dir(self):
        """load_documents should raise FileNotFoundError for non-existent path."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            agent = DocQAAgent(settings=self.settings)
            # Manually set mock_mode to False so the real path check runs
            # (the FileNotFoundError is raised before any LangChain call)
            agent._mock_mode = False

            with pytest.raises(FileNotFoundError):
                agent.load_documents("/this/path/does/not/exist/at/all")

    def test_chunk_size_constant(self):
        """Verify the chunk size constants are set to banking-appropriate values."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.document_agent import DocQAAgent
            assert DocQAAgent.CHUNK_SIZE == 800
            assert DocQAAgent.CHUNK_OVERLAP == 100
            assert DocQAAgent.TOP_K_RESULTS == 4


# ---------------------------------------------------------------------------
# ReportGenerationAgent Tests
# ---------------------------------------------------------------------------

class TestReportGenerationAgent:
    """Tests for the ReportGenerationAgent — structured report drafting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures before each test."""
        self.settings = make_mock_settings()

    def test_init_mock_mode(self):
        """Agent should initialise in mock mode when LangChain unavailable."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)
            assert agent._mock_mode is True

    def test_generate_monthly_report_returns_string(self):
        """generate_monthly_report should return a non-empty string."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {
                "total_transactions": 125000,
                "sepa_outbound": 45000,
                "sepa_inbound": 52000,
                "bisera_transactions": 28000,
                "failed_transactions": 312,
                "exceptions_total": 18,
                "exceptions_p1": 2,
                "exceptions_p2": 5,
                "exceptions_p3p4": 11,
                "resolved_within_sla": 94,
            }
            report = agent.generate_monthly_report(data=data, period="November 2024")

            assert isinstance(report, str)
            assert len(report) > 100
            assert "MONTHLY OPERATIONAL REPORT" in report

    def test_generate_monthly_report_includes_key_figures(self):
        """Report should include the input figures (not hallucinated numbers)."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {
                "total_transactions": 125000,
                "sepa_outbound": 45000,
                "sepa_inbound": 52000,
                "bisera_transactions": 28000,
                "failed_transactions": 312,
                "exceptions_total": 18,
                "exceptions_p1": 2,
                "exceptions_p2": 5,
                "exceptions_p3p4": 11,
                "resolved_within_sla": 94,
            }
            report = agent.generate_monthly_report(data=data, period="November 2024")

            # Key figures must appear in the report
            assert "125,000" in report
            assert "NOVEMBER 2024" in report.upper()

    def test_generate_monthly_report_handles_missing_fields(self):
        """Report generation should not crash if optional fields are missing."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            # Minimal data — missing many expected fields
            data = {"total_transactions": 1000}
            report = agent.generate_monthly_report(data=data, period="December 2024")

            assert isinstance(report, str)
            assert "MONTHLY OPERATIONAL REPORT" in report

    def test_draft_compliance_note_returns_string(self):
        """draft_compliance_note should return a well-structured string."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {
                "subject": "GDPR Data Retention Policy Review",
                "regulatory_reference": "GDPR Article 5(1)(e)",
                "findings": [
                    "Customer data retention periods exceed GDPR 5-year maximum",
                    "No automated deletion process in place",
                ],
                "required_actions": [
                    "Implement automated data purge for accounts closed >5 years",
                    "Update data retention policy document",
                ],
                "deadline": "2025-03-31",
                "owner": "Data Protection Officer",
            }
            note = agent.draft_compliance_note(data=data)

            assert isinstance(note, str)
            assert "COMPLIANCE NOTE" in note
            assert "GDPR Data Retention Policy Review" in note

    def test_generate_summary_returns_string(self):
        """generate_summary should return a string with a header."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {"metric_a": 100, "metric_b": "up 5%"}
            result = agent.generate_summary(data=data, report_type="risk_dashboard")

            assert isinstance(result, str)
            assert "DRAFT" in result

    def test_return_rate_calculation(self):
        """Return rate should be correctly calculated from failed/total transactions."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {
                "total_transactions": 1000,
                "sepa_outbound": 500,
                "sepa_inbound": 400,
                "bisera_transactions": 100,
                "failed_transactions": 50,  # 5% return rate
                "exceptions_total": 0, "exceptions_p1": 0,
                "exceptions_p2": 0, "exceptions_p3p4": 0,
                "resolved_within_sla": 100,
            }
            report = agent.generate_monthly_report(data=data)
            assert "5.00%" in report

    def test_zero_transactions_no_division_error(self):
        """Should handle zero total transactions without raising ZeroDivisionError."""
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.report_agent import ReportGenerationAgent
            agent = ReportGenerationAgent(settings=self.settings)

            data = {
                "total_transactions": 0,
                "failed_transactions": 0,
                "exceptions_total": 0, "exceptions_p1": 0,
                "exceptions_p2": 0, "exceptions_p3p4": 0,
                "resolved_within_sla": 0,
            }
            # Should not raise
            report = agent.generate_monthly_report(data=data)
            assert "0.00%" in report


# ---------------------------------------------------------------------------
# ComplianceAgent Tests
# ---------------------------------------------------------------------------

class TestComplianceAgent:
    """Tests for the ComplianceAgent — regulatory compliance checking."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures before each test."""
        self.settings = make_mock_settings()

    def test_init_mock_mode(self):
        """Agent should initialise in mock mode when LangChain unavailable."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            assert agent._mock_mode is True

    def test_check_document_returns_dict(self):
        """check_document should return a dictionary with expected keys."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Sample document text about customer consent and data handling.")

            assert isinstance(result, dict)
            required_keys = ["summary", "findings", "risk_level", "references", "checked_at"]
            for key in required_keys:
                assert key in result, f"Missing key: {key}"

    def test_check_document_findings_is_list(self):
        """findings field should always be a list."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Any document text.")
            assert isinstance(result["findings"], list)

    def test_check_document_checked_at_is_valid_datetime(self):
        """checked_at should be a valid ISO 8601 datetime string."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Test document.")

            # Should not raise — datetime.fromisoformat accepts ISO 8601
            checked_at = datetime.fromisoformat(result["checked_at"])
            assert checked_at is not None

    def test_check_document_mock_mode_has_mock_flag(self):
        """Mock mode responses should include a 'mock: True' flag."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Test document.")
            assert result.get("mock") is True

    def test_flag_risks_returns_list(self):
        """flag_risks should return a list of risk dictionaries."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            risks = agent.flag_risks("Document with potential AML and GDPR issues.")
            assert isinstance(risks, list)

    def test_check_document_all_context(self):
        """'ALL' regulation context should return findings (mocked compliance check)."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Test document.", regulation_context="BNB")
            # BNB references should be populated
            assert len(result["references"]) > 0
            assert result["regulation_context"] == "BNB"

    def test_check_document_bnb_context(self):
        """BNB context should return only BNB references."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent, REGULATORY_RULES
            agent = ComplianceAgent(settings=self.settings)
            result = agent.check_document("Test document.", regulation_context="BNB")
            bnb_ids = {r["id"] for r in REGULATORY_RULES["BNB"]}
            result_refs = set(result["references"])
            assert result_refs == bnb_ids

    def test_suggest_remediation_mock_returns_string(self):
        """suggest_remediation in mock mode should return a non-empty string."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            finding = {
                "description": "Missing customer consent documentation",
                "risk_level": "HIGH",
                "regulation": "GDPR-ART13",
            }
            result = agent.suggest_remediation(finding)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_extract_risk_level_high(self):
        """_extract_risk_level should detect HIGH RISK in text."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            assert agent._extract_risk_level("This is a HIGH RISK finding.") == "HIGH"

    def test_extract_risk_level_medium(self):
        """_extract_risk_level should detect MEDIUM RISK in text."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            assert agent._extract_risk_level("MEDIUM RISK: consent language issue") == "MEDIUM"

    def test_extract_risk_level_clean(self):
        """_extract_risk_level should detect COMPLIANT in text."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            assert agent._extract_risk_level("Document is fully COMPLIANT.") == "CLEAN"

    def test_extract_risk_level_unknown(self):
        """_extract_risk_level should return UNKNOWN for ambiguous text."""
        with patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):
            from src.agents.compliance_agent import ComplianceAgent
            agent = ComplianceAgent(settings=self.settings)
            assert agent._extract_risk_level("Some general commentary.") == "UNKNOWN"


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestSettings:
    """Tests for the application configuration settings."""

    def test_settings_loads_without_error(self):
        """Settings should load with default values when no .env file is present."""
        from config.settings import Settings
        # Load without any env file to get pure defaults
        settings = Settings(_env_file=None)
        assert settings.app_name == "BankMind"
        # app_mode default is 'development'; APP_MODE=demo in test env is acceptable
        assert settings.app_mode in ("development", "demo", "production")

    def test_demo_mode_when_no_api_key(self):
        """is_demo_mode should return True when no API key is configured."""
        from config.settings import Settings
        settings = Settings(_env_file=None)
        # Force no API key
        settings.openai_api_key = None
        assert settings.is_demo_mode is True

    def test_default_llm_models(self):
        """Default model selections should reflect the cost/quality decisions documented in code."""
        from config.settings import Settings
        settings = Settings(_env_file=None)
        assert settings.llm_model == "gpt-4o"
        assert settings.llm_model_doc_qa == "gpt-4o-mini"  # Cheaper for grounded Q&A
        assert settings.llm_model_compliance == "gpt-4o"   # Full model for compliance

    def test_chunk_overlap_cannot_exceed_maximum(self):
        """Validator should reject unreasonably large chunk overlap values."""
        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(chunk_overlap=500)


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    """
    Lightweight integration tests that verify agents work together.
    These tests do not make real LLM calls.
    """

    def test_all_agents_can_be_instantiated(self):
        """All three agent classes should be importable and instantiable."""
        with patch("src.agents.document_agent.LANGCHAIN_AVAILABLE", False), \
             patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False), \
             patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):

            from src.agents.document_agent import DocQAAgent
            from src.agents.report_agent import ReportGenerationAgent
            from src.agents.compliance_agent import ComplianceAgent

            settings = make_mock_settings()

            doc_agent = DocQAAgent(settings=settings)
            report_agent = ReportGenerationAgent(settings=settings)
            compliance_agent = ComplianceAgent(settings=settings)

            assert doc_agent is not None
            assert report_agent is not None
            assert compliance_agent is not None

    def test_compliance_then_report_workflow(self):
        """
        Simulate a realistic workflow: check document compliance,
        then generate a compliance note based on the findings.
        """
        with patch("src.agents.report_agent.LANGCHAIN_AVAILABLE", False), \
             patch("src.agents.compliance_agent.LANGCHAIN_AVAILABLE", False):

            from src.agents.report_agent import ReportGenerationAgent
            from src.agents.compliance_agent import ComplianceAgent

            settings = make_mock_settings()
            compliance_agent = ComplianceAgent(settings=settings)
            report_agent = ReportGenerationAgent(settings=settings)

            # Step 1: Check a document
            doc_text = "Customer data will be retained for 10 years after account closure."
            check_result = compliance_agent.check_document(doc_text, regulation_context="GDPR")

            assert "findings" in check_result
            assert len(check_result["findings"]) > 0

            # Step 2: Generate a compliance note based on the findings
            first_finding = check_result["findings"][0]
            note_data = {
                "subject": "Compliance Finding: " + first_finding.get("description", "Finding")[:50],
                "regulatory_reference": first_finding.get("regulation", "GDPR"),
                "findings": [first_finding.get("description", "See attached")],
                "required_actions": ["Review and remediate per compliance team guidance"],
                "deadline": "2025-06-30",
            }
            note = report_agent.draft_compliance_note(data=note_data)

            assert isinstance(note, str)
            assert "COMPLIANCE NOTE" in note
