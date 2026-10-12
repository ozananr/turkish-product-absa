import pandas as pd

from absa.data.masker import mask_personal_data


def clean_reviews(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Cleans a dataframe of reviews based on length, duplicates, and templates."""
    # 1. Birebir aynı olan yorumları (duplicate) sil
    df_cleaned = df.drop_duplicates(subset=[text_col]).copy()

    # 2. Eksik (NaN) metinleri sil
    df_cleaned = df_cleaned.dropna(subset=[text_col])

    # 3. Kişisel verileri maskele
    df_cleaned[text_col] = df_cleaned[text_col].apply(mask_personal_data)

    # 4. 3 kelimeden kısa olan yorumları filtrele
    df_cleaned = df_cleaned[df_cleaned[text_col].str.split().str.len() >= 3]

    # 5. Anlamsız/şablon yorumları filtrele (büyük/küçük harf duyarsız)
    templates = ["hızlı kargo", "güzel ürün", "teşekkürler", "tavsiye ederim", "çok güzel"]
    pattern = "|".join([f"^{t}$" for t in templates])
    df_cleaned = df_cleaned[~df_cleaned[text_col].str.lower().str.strip().str.fullmatch(pattern)]

    return df_cleaned
