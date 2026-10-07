import re

def clean(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Removing links
    text = re.sub(r'[@#]\w+', '', text) # Removing tags
    # Since some words change their meaning when repeated (such as "bitte" or "dass")
    # it is best to consider only repetitions of three or more occurrences
    text = re.sub(r'(\w)\1{2,}', r'\1', text) # We also remove repeated letters (e.g., converting "langsaaaaaam" to "langsam").
    text = re.sub(r'\s+', ' ', text).strip() # Remove empty lines that may appear after cleaning.
    return text