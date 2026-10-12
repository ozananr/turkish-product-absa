import re


def mask_personal_data(text: str) -> str:
    """Masks email addresses and phone numbers in a given text."""
    if not isinstance(text, str):
        return str(text)

    # E-posta maskeleme
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    text = re.sub(email_pattern, "[EPOSTA]", text)

    # Telefon numarası maskeleme (TR formatları)
    phone_pattern = r"(?:\+90|0)?\s*[5]\d{2}\s*\d{3}\s*\d{2}\s*\d{2}"
    text = re.sub(phone_pattern, "[TELEFON]", text)

    return text
