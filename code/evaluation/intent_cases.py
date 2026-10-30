"""Custom intent probes: requests with a known label, for testing the intent classifier.

Each probe is hand-written (not from the dataset). `deletion` is the expected label.
"""
from typing import List, NamedTuple


class Probe(NamedTuple):
    text: str
    deletion: bool
    language: str
    note: str


PROBES = [
    Probe(
        "Please remove me from all of your marketing lists. I still want to keep shopping with my account.",
        False,
        "en",
        "marketing opt-out that contains the deletion verb 'remove'",
    ),
    Probe(
        "My surname is spelled wrong on my profile. Please change Smyth to Smith.",
        False,
        "en",
        "surname correction",
    ),
    Probe(
        "Please send me a copy of all the personal data you hold about me. Don't delete anything, I only want to see it.",
        False,
        "en",
        "request for a data copy with an explicit 'don't delete'",
    ),
    Probe(
        "Bitte löschen Sie mein Konto und alle meine personenbezogenen Daten dauerhaft.",
        True,
        "de",
        "German deletion request (language the keyword fallback does not cover)",
    ),
    Probe(
        "Proszę o usunięcie mojego konta oraz wszystkich moich danych osobowych.",
        True,
        "pl",
        "Polish deletion request (language the keyword fallback does not cover)",
    ),
    Probe(
        "私のデータをダウンロードしたいです。アカウントは削除しないでください。",
        False,
        "ja",
        "Japanese request to download data without deleting",
    ),
    Probe(
        "Ma ville de livraison est incorrecte, merci de la corriger. Ne supprimez rien, je garde mon compte.",
        False,
        "fr",
        "French correction that says 'ne supprimez rien'",
    ),
    Probe(
        "Please delete my old shipping address from my profile, but keep my account active.",
        False,
        "en",
        "removing one old address while keeping the account is a correction",
    ),
    Probe(
        "Please close my account permanently and erase everything you have on me.",
        True,
        "en",
        "plain deletion request",
    ),
    Probe(
        "I can't log in to my account. Please help me get access again; do not close the account.",
        False,
        "en",
        "login help with 'do not close the account'",
    ),
]  # type: List[Probe]
