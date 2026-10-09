"""Wrong-script check for final replies.

Every task is in English or Banglish, so a correct reply uses only Latin
letters. A reply counts as wrong-script when it contains any non-Latin letter or
non-ASCII digit (Bangla script, Chinese, and so on). Symbols such as the taka
sign are not letters and do not count. This is reported alongside the score; it
does not change success.
"""

import unicodedata

LATIN_SCRIPT_LANGUAGES = ("en", "banglish")
# Non-ASCII letters that are not named LATIN but are normal in Latin text.
LATIN_SAFE = set("µªº")


def non_latin_scripts(text):
    """Scripts of the non-Latin letters and digits in text, for example {'BENGALI', 'CJK'}."""
    scripts = set()
    for char in text or "":
        if char.isascii() or char in LATIN_SAFE:
            continue
        category = unicodedata.category(char)
        if not (category.startswith("L") or category == "Nd"):
            continue
        name = unicodedata.name(char, "UNKNOWN")
        if "LATIN" not in name:
            scripts.add(name.split()[0])
    return scripts


def is_wrong_script(language, reply):
    return language in LATIN_SCRIPT_LANGUAGES and bool(non_latin_scripts(reply))
