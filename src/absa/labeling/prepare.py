import pandas as pd

from absa.schema import ASPECTS, LABELS

# C:\ klasöründeki dosyaları okurken \r ve \l hatası almamak için / kullanıyoruz:
sample = pd.read_csv("C:/reviews_clean.csv")
pre = pd.read_csv("C:/labeling_sample.csv")

df = sample.merge(pre, on="review_id", how="inner")
print(f"Satır Sayıları -> Sample: {len(sample)}, Pre: {len(pre)}, Merge: {len(df)}")

print("\n--- Etiket Kontrolü ---")
for a in ASPECTS:
    bad = df[~df[a].isin(LABELS)]
    if len(bad) > 0:
        print(f"Uyarı: {a} sütununda geçersiz etiket var: {len(bad)} adet")
    else:
        print(f"Temiz: {a} sütunu")

# Sonucu nereye kaydetmek istiyorsanız orayı belirtiyoruz
# Yine / kullanarak C'nin içine kaydediyoruz
df.to_csv("C:/master_prelabeled.csv", index=False)
print("\nİşlem tamamlandı. 'C:/master_prelabeled.csv' oluşturuldu.")
