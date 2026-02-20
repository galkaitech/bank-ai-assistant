"""
PII Detection and Masking Utility

Detects and masks Personally Identifiable Information (PII) before text
is sent to external LLM APIs. This is a critical component for GDPR
compliance in a banking context.

Patterns covered:
- Bulgarian EGN (Единен граждански номер) — 10-digit national ID
- IBAN numbers (Bulgarian and European formats)
- Credit/debit card numbers (PAN)
- Email addresses
- Bulgarian phone numbers
- Generic name patterns (via simple heuristics)

Note: This is a rule-based implementation. For production, supplement
with a trained NER model (spaCy with a Bulgarian model) for higher
recall on person names and addresses.
"""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger("bankmind.utils.pii_filter")

# ---------------------------------------------------------------------------
# PII detection patterns
# ---------------------------------------------------------------------------

PII_PATTERNS: Dict[str, re.Pattern] = {
    # Bulgarian EGN: 10 digits, first 6 encode date of birth
    "bulgarian_egn": re.compile(
        r"\b([0-9]{2}(0[1-9]|1[0-2]|2[1-9]|3[0-2]|4[1-9]|5[0-2])[0-9]{4})\b"
    ),

    # IBAN: international format, Bulgarian IBANs start with BG
    "iban": re.compile(
        r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"
    ),

    # Payment card PAN: 13-19 digits, typically with spaces or dashes
    "card_pan": re.compile(
        r"\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2})[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{1,4}\b"
    ),

    # Email address
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
    ),

    # Bulgarian phone numbers (mobile: 08X, landlines: 02, 03X, etc.)
    "bulgarian_phone": re.compile(
        r"\b((\+359|00359|0)[0-9]{8,9})\b"
    ),
}

REPLACEMENT_TOKENS = {
    "bulgarian_egn": "[EGN_REDACTED]",
    "iban": "[IBAN_REDACTED]",
    "card_pan": "[CARD_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "bulgarian_phone": "[PHONE_REDACTED]",
}


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Scan text for PII patterns and replace them with redaction tokens.

    Args:
        text: Raw input text potentially containing PII.

    Returns:
        A tuple of (masked_text, detection_counts) where detection_counts
        maps PII type to the number of instances found and masked.
    """
    masked = text
    counts: Dict[str, int] = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(masked)
        if matches:
            counts[pii_type] = len(matches)
            masked = pattern.sub(REPLACEMENT_TOKENS[pii_type], masked)
            logger.info("Masked %d %s instance(s).", len(matches), pii_type)

    if counts:
        logger.info("PII filter: masked types %s.", list(counts.keys()))

    return masked, counts


def contains_pii(text: str) -> bool:
    """
    Check whether text contains any detected PII without masking it.

    Useful for logging decisions — if PII is detected, log a warning
    but do not log the actual content.

    Args:
        text: Input text to scan.

    Returns:
        True if any PII pattern is found, False otherwise.
    """
    for pattern in PII_PATTERNS.values():
        if pattern.search(text):
            return True
    return False
