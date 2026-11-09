"""Component-wise attribution for scientific claim verification on SciFact."""

__version__ = "0.1.0"

LABELS = ("SUPPORT", "REFUTE", "NEI")
ABSTAIN = "ABSTAIN"

# SciFact's released JSONL writes the refuting class as "CONTRACT".
# It is the same class; the mapping lives here so no other module has to know about it.
LABEL_ALIASES = {
    "SUPPORT": "SUPPORT",
    "CONTRACT": "REFUTE",
    "REFUTE": "REFUTE",
    "NOT_ENOUGH_INFO": "NEI",
    "NOT_ENOUGHINFORMATION": "NEI",
    "NEI": "NEI",
}
