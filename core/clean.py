import re

def clean_text(text):
    """Remove URLs, tags, and 3+ character repetitions; collapse whitespace."""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)

    # Some words change meaning when letters are collapsed (e.g., "bitte", "dass"),
    # so only repetitions of 3+ are removed.
    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text