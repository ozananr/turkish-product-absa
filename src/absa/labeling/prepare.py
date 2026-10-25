import numpy as np
import pandas as pd

from absa.schema import ASPECTS, LABELS

# C:\ klasöründeki dosyaları okurken \r ve \l hatası almamak için / kullanıyoruz:
sample = pd.read_csv("C:/reviews_clean.csv")
pre = pd.read_csv("C:/prelabels.csv")

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
df.to_csv("C:/Users/yusuf/OneDrive/Desktop/Medikalmaster_prelabeled.csv", index=False)
print("\nİşlem tamamlandı. 'master_prelabeled.csv' oluşturuldu.")

# --- ADIM 3: KAPPA KÜMESİNİ AYIRMA ---
print("\n--- Kappa Kümesi Hazırlanıyor ---")

kappa = df.sample(200, random_state=42)
rest = df.drop(kappa.index)

# BURASI DÜZELTİLDİ: 'text_x' olarak alıp, adını tekrar 'text' yapıyoruz
blank = kappa[["review_id", "text_x"]].copy()
blank = blank.rename(columns={"text_x": "text"})

for a in ASPECTS:
    blank[a] = ""  # Ön etiket YOK

blank.to_csv("kappa_set_blank.csv", index=False)
kappa.to_csv("kappa_set_with_prelabels.csv", index=False)
rest.to_csv("remaining_prelabeled.csv", index=False)
# --- ADIM 4: KALAN YORUMLARI ÜÇE BÖLME ---
print("\n--- Kalan Yorumlar 3'e Bölünüyor ---")

# Önce 'rest' (kalanlar) tablosunu da temizleyelim
if "text_x" in rest.columns:
    rest = rest.rename(columns={"text_x": "text"})
if "text_y" in rest.columns:
    rest = rest.drop(columns=["text_y"])  # Fazlalık sütunu sil

# Veriyi tamamen rastgele karıştırıyoruz
rest = rest.sample(frac=1, random_state=42)

# --- DÜZELTİLEN KISIM BURASI ---
# Doğrudan tabloyu değil, satır numaralarını 3'e bölüyoruz
indices = np.array_split(rest.index, 3)
parts = [rest.loc[idx] for idx in indices]
# -------------------------------

# Dokümantasyon (README vb.) için tablo hazırlığı
print("\n📝 DOKÜMANTASYONA EKLENECEK TABLO:")
print("| Etiketleyici | Dosya Adı | Yorum Sayısı | Örnek ID Aralığı (Min-Max) |")
print("|---|---|---|---|")

toplam_yorum = 0
all_ids = set()

for i, part in enumerate(parts, start=1):
    dosya_adi = f"batch_person{i}.csv"
    part.to_csv(dosya_adi, index=False)  # Dosyayı kaydet

    kisi_yorum_sayisi = len(part)
    toplam_yorum += kisi_yorum_sayisi

    # Kesişim kontrolü için benzersiz ID'leri topluyoruz
    all_ids.update(part["review_id"].tolist())

    min_id = part["review_id"].min()
    max_id = part["review_id"].max()

    # Terminale Markdown formatında tablo satırını basıyoruz
    print(f"| Kişi {i} | `{dosya_adi}` | {kisi_yorum_sayisi} | {min_id} - {max_id} |")

# Sağlamlama (Kontrol) Testleri
print("\n--- Güvenlik Kontrolü ---")
if toplam_yorum == len(rest):
    print(f"✅ Matematik doğru: Parçaların toplamı ({toplam_yorum}) kalan yoruma eşit.")
else:
    print(
        f"❌ HATA: Yorum sayıları tutmuyor! (Toplam: {toplam_yorum}, Olması Gereken: {len(rest)})"
    )

if len(all_ids) == len(rest):
    print("✅ Hiçbir yorum birden fazla kişiye gitmedi (Kesişim yok).")
else:
    print("❌ HATA: Bazı yorumlar farklı kişilere aynı anda gitmiş!")
