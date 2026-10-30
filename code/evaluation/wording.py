"""Small helpers for the sentences the evaluation scripts and reports generate."""
from typing import Optional


def plural(count: int, noun: str, plural_form: Optional[str] = None) -> str:
    """'1 step', '2 steps', '0 steps'. Pass plural_form for an irregular noun: plural(2, 'entry', 'entries')."""
    return "%d %s" % (count, noun if count == 1 else (plural_form or noun + "s"))
