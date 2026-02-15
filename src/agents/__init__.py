"""
BankMind agent modules.

Each agent is responsible for a specific domain of banking tasks:
- DocQAAgent: Retrieval-Augmented Generation over internal documents
- ReportGenerationAgent: Structured report and summary drafting
- ComplianceAgent: Regulatory compliance checking and risk flagging
"""

from src.agents.document_agent import DocQAAgent
from src.agents.report_agent import ReportGenerationAgent
from src.agents.compliance_agent import ComplianceAgent

__all__ = ["DocQAAgent", "ReportGenerationAgent", "ComplianceAgent"]
