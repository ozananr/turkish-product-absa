# Etiketleme Örneği (Labeling Sample)

Temiz yorum havuzundan (`data/interim/reviews_clean.csv`, 220.985 yorum) etiketleme için seçilen dengeli örnek.
Amaç: etiketleme emeğinin modele en çok şey öğreten yorumlara gitmesi.

## Yeniden üretme

```bash
uv run python -m absa.data.sampling --pool data/interim/reviews_clean.csv --n 2500 --seed 42
```

Çıktılar `data/interim/` altına yazılır ve git'e eklenmez:

| Dosya | İçerik |
|---|---|
| `labeling_sample.csv` | Örnek. Kolonlar: `review_id, text, source, label, rating, product_name, category, hit_kargo_paketleme, hit_satici_hizmet, hit_beden_uyum` |
| `labeling_sample_stats.json` / `.md` | İstatistikler |

Aynı havuz ve aynı seed her zaman aynı örneği verir (havuz önce `review_id`'ye göre sıralanır; iki çalıştırmanın SHA256 özeti aynıdır).
`review_id` havuzdan olduğu gibi korunur (`trsa_<id>`, `trendyol_<satır>`), bu yüzden kararlıdır.
Kod: `src/absa/data/sampling.py`, `keywords.py`, `categories.py`. Testler: `tests/test_sampling.py`, `test_keywords.py`, `test_categories.py`.

## Seçim stratejisi

1. **Boyut:** 2.500 yorum, `seed=42`.
2. **Hücre kotaları (kaynak x etiket):**
   - Kaynak: Trendyol %40 (1.000), TRSAv1 %60 (1.500). Trendyol'da negatif/nötr az olduğu için çoğu negatif yorum TRSAv1'den gelir.
   - Etiket: negatif %40, nötr %30, pozitif %30. Havuzda oranlar %22 / %21 / %56 idi, yani negatif ve nötr fazla örneklendi.
   - Etiket: TRSAv1'de kaynak etiketi, Trendyol'da puandan türetilen etiket (1-2 negatif, 3 nötr, 4-5 pozitif).
3. **Nadir aspect kotası:** Her hücrede, her nadir aspect için en az %15 anahtar kelime isabeti hedeflenir.
4. **Kategori dengesi:** Her hücrede bilinen bir kategori en fazla %30 pay alır. Kategori yalnızca ürün adı olan (Trendyol) yorumlarda bilinir. TRSAv1 satırlarında ürün adı olmadığı için kategori `bilinmiyor` olur ve sınıra dahil edilmez.
5. **Ürün sınırı:** Aynı üründen en fazla 10 yorum seçilir (tek ürünün örneği domine etmesini önler).

## Anahtar kelime listesi (nadir aspect'ler)

Liste kodda `src/absa/data/keywords.py` içindedir. Metin Türkçe küçük harfe çevrilir.
`prefix` kelime başında eşleşir (ek alabilir: "kargo" -> "kargoyu"), `exact` tam kelime olarak eşleşir ("dar" -> "darbe" eşleşmez).

> Bu liste yalnızca **aday** seçmek içindir. Anahtar kelimenin geçmesi aspect'in yorumda olduğunu garanti etmez; aspect etiketini insan verir.

| Aspect | Başlangıç eşleşmeli (`prefix`) | Tam kelime (`exact`) |
|---|---|---|
| `kargo_paketleme` | kargo, paket, ambalaj, kutu, koli, teslimat, teslim, kurye, gönderi, sevkiyat, yurtiçi, mng, ptt, hasarlı, ezik, ezilmiş, balonlu, poşet | geç geldi, geç ulaştı, geç teslim |
| `satici_hizmet` | satıcı, mağaza, iade, değişim, müşteri hizmet, müşteri temsilci, iletişim, garanti, servis, geri ödeme, şikayet, destek, cevap, ilgilen, yanıt, mesaj | firma |
| `beden_uyum` | beden, kalıp, numara, ölçü, büyük geldi, küçük geldi, büyük aldım, küçük aldım, bir beden, sıkıyor, sıktı, boyu, uydu, uyuyor, oturdu, oturuyor, rahat, tam oldu | dar, bol, xl, xxl, xs, s beden, boy |

(Tablo özetlenmiştir; tam liste ve yazım varyantları kodda.)

## Kategori kuralları

Kategori, ürün adındaki kelimelerden kural tabanlı çıkarılır. Kurallar sırayla denenir, ilk eşleşen seçilir:
`hijyen_ped` -> `ayakkabi` -> `spor` -> `aksesuar` -> `kozmetik_bakim` -> `giyim` -> `diger`.
Sıra önemlidir (örnek: "Emici Külot Lohusa Pedi" giyim değil `hijyen_ped`). Ürün adı olmayan satırlar `bilinmiyor` olur.

## Örnek istatistikleri (seed=42)

- **Örnek boyutu:** 2.500 (hedef 2.500, benzersiz `review_id`: 2.500)
- **Kaynak dağılımı:** TRSAv1 1.500, Trendyol 1.000

**Etiket dağılımı**

| Etiket | Toplam | Trendyol | TRSAv1 |
|---|---:|---:|---:|
| negative | 1.000 (%40,0) | 400 | 600 |
| neutral | 750 (%30,0) | 300 | 450 |
| positive | 750 (%30,0) | 300 | 450 |

**Trendyol puan dağılımı:** 1★ 262, 2★ 138, 3★ 300, 4★ 37, 5★ 263

**Kategori dağılımı**

| Kategori | Yorum | Bilinen kategoriler içinde pay |
|---|---:|---:|
| bilinmiyor (TRSAv1) | 1.500 | - |
| kozmetik_bakim | 300 | %30,0 |
| giyim | 300 | %30,0 |
| diger | 119 | %11,9 |
| aksesuar | 118 | %11,8 |
| ayakkabi | 77 | %7,7 |
| hijyen_ped | 43 | %4,3 |
| spor | 43 | %4,3 |

Bilinen kategorilerde en büyük pay %30,0 (sınır korundu). Trendyol örneğinde 389 benzersiz ürün var.

**Nadir aspect anahtar kelime isabetleri**

| Aspect | İsabet | Pay |
|---|---:|---:|
| `kargo_paketleme` | 448 | %17,9 |
| `satici_hizmet` | 405 | %16,2 |
| `beden_uyum` | 470 | %18,8 |

En az bir nadir aspect isabeti olan yorum: 1.095 (%43,8).

## Sınırlar

- Kategori yalnızca Trendyol yorumlarında bilinir (TRSAv1'de ürün adı yok). Kategori dengesi bu yüzden yalnızca Trendyol payına uygulanır; TRSAv1 yorumları (%60) kategori sınırının dışındadır.
- Kategori kuralları ve anahtar kelime listesi sezgiseldir; `diger` kategorisi (%11,9) kuralların yakalamadığı ürünleri içerir.
- Trendyol'da 4★ yorum az (37), çünkü etiket hedefi negatif ve nötr ağırlıklıdır (4-5★ pozitif sayılır ve pozitif kota %30'dur).
- Örnek dosyası repoda değildir, Drive'da paylaşılır (bkz. `data/README.md`).
