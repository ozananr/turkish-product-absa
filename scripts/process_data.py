import os

import pandas as pd

from absa.data.cleaner import clean_reviews


def main():
    raw_dir = "data/raw"
    interim_dir = "data/interim"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(interim_dir, exist_ok=True)

    print("Pipeline testi için 15.000 satırlık temsili (dummy) veri seti oluşturuluyor...")
    # 10.000'i temiz çıkacak şekilde tasarlanmış veri seti
    data = {
        "review_id": range(15000),
        "source": ["trendyol"] * 15000,
        "text": ["Kamera muazzam ama şarjı hemen bitiyor. İletişim: 0555 555 55 55"] * 5000
        + ["güzel ürün"] * 5000
        + ["Fiyat performans ürünü kesinlikle tavsiye ederim, kargo da çok hızlıydı."] * 5000,
        "label": ["positive"] * 10000 + ["neutral"] * 5000,
        "product_name": ["Telefon"] * 15000,
    }
    df_combined = pd.DataFrame(data)

    initial_count = len(df_combined)
    print(f"Başlangıçtaki toplam yorum sayısı: {initial_count}")

    # Temizleme ve Maskeleme
    df_cleaned = clean_reviews(df_combined, text_col="text")
    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    print(f"Temizleme sonrası kalan yorum sayısı: {final_count}")
    print(f"Silinen kopya/şablon yorum sayısı: {dropped_count}")
    print(f"Elenme oranı: {dropped_count / initial_count:.2%}")

    # Temizlenmiş veriyi kaydet
    cleaned_path = f"{interim_dir}/cleaned_reviews.csv"
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"\nBaşarılı! Temizlenmiş veri şuraya kaydedildi: {cleaned_path}")


if __name__ == "__main__":
    main()
