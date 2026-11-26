"""Text normalisation shared by training and inference.

Keeping a single function here guarantees the web app sees exactly the same
preprocessing as the model did during training (the v1 app did not).
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_LEET = str.maketrans({"@": "a", "$": "s", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_NON_ALPHA = re.compile(r"[^a-z\s']")
_REPEAT = re.compile(r"(.)\1{2,}")  # "stuuuupid" -> "stuupid"
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9@$*']+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    # undo common obfuscation inside words (f*ck, b1tch, $tupid)
    t = " ".join(w.translate(_LEET).replace("*", "u") if any(c.isalpha() for c in w) else w
                 for w in t.split())
    t = _NON_ALPHA.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)
    return _SPACES.sub(" ", t).strip()


def tokenize_display(text: str) -> list[str]:
    """Split raw text into word tokens for explanation/highlighting."""
    return _WORD.findall(text or "")
