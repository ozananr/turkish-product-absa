import pandas as pd

from absa.data.cleaner import clean_reviews
from absa.data.masker import mask_personal_data


def test_mask_personal_data():
    phone_text = "Bana 0555 555 55 55 numarasından ulaşın."
    assert mask_personal_data(phone_text) == "Bana [TELEFON] numarasından ulaşın."

    assert mask_personal_data("Mail adresim test@example.com") == "Mail adresim [EPOSTA]"
    assert mask_personal_data("Normal bir yorum.") == "Normal bir yorum."


def test_clean_reviews():
    data = {
        "review_id": [1, 2, 3, 4, 5],
        "text": [
            "Kamera muazzam ama şarjı hemen bitiyor.",  # Kalmalı
            "güzel ürün",  # Silinmeli (Şablon)
            "İyi",  # Silinmeli (<3 kelime)
            "Kamera muazzam ama şarjı hemen bitiyor.",  # Silinmeli (Tekrar/Duplicate)
            "Ürün bozuk ahmet@gmail.com adresinden ulaşın.",  # Kalmalı ama e-posta maskelenmeli
        ],
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_reviews(df)

    # Beklenen: 2 satır kalmalı (1. ve 5. satırlar)
    assert len(cleaned_df) == 2

    # Beklenen: 5. satırdaki e-posta maskelenmiş olmalı
    assert "[EPOSTA]" in cleaned_df.iloc[1]["text"]
