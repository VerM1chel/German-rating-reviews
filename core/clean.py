import re

def clean(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Убираем ссылки
    text = re.sub(r'[@#]\w+', '', text) # Убираем теги
    # Поскольку есть слова, которые меняют смысл при повторах "bitte", "dass"
    # Лучши учитывать повторы только от повторений 3 и более раз
    text = re.sub(r'(\w)\1{2,}', r'\1', text) # Убираем повторы букв (например, "langsaaaaaam" в "langsam")
    text = re.sub(r'\s+', ' ', text).strip() # Убираем пустые строки, которые могут появиться после очистки
    return text