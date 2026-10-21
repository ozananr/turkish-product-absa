# LLM ile Ön Etiketleme (Pre-labeling)

Etiketleme örneği (`data/interim/labeling_sample.csv`, 2.500 yorum) bir LLM ile ön etiketlenir.
Etiketleyiciler sıfırdan etiketlemek yerine önerileri düzeltir. Kappa seti (200 yorum) etiketleyicilere
ön etiketsiz gösterilir, ama LLM'i insanlarla karşılaştırmak için o da ön etiketlenir.

Kod: `src/absa/labeling/`. Prompt: `src/absa/labeling/prompts/v1.md`. Testler: `tests/test_labeling_*.py`.

## Sağlayıcı ve model seçimi

**Seçim:** Google Gemini API, **ücretli tier**, model `gemini-3.5-flash-lite`.
Kod OpenAI uyumlu Chat Completions API kullanır, bu yüzden sağlayıcı ya da model yalnızca `.env` ile değişir.

| Kriter | Değerlendirme |
|---|---|
| Maliyet | `gemini-3.5-flash-lite`: 1M girdi token $0,30, önbellekli girdi $0,03, çıktı $2,50 (düşünme token'ları dahil). Prompt yaklaşık 2.000 token, cevap yaklaşık 80 token. 2.500 yorum için tahmin: 5,3M girdi + 0,2M çıktı ≈ **2 USD** (önbelleksiz üst sınır). 50 yorumluk pilot birkaç sent. |
| Veri gizliliği | Ücretsiz tier'da Google girdileri ve çıktıları ürünlerini geliştirmek için kullanabilir. Kaynak veri setleri lisanslı ve erişimi kısıtlı olduğu için **ücretli tier** kullanılır (ücretli tier verisi eğitimde kullanılmaz). |
| Rate limit | Ücretli tier'da limit proje bazındadır ve AI Studio'da görünür. Script 429 hatalarında bekleyip tekrar dener (`openai` SDK, 6 deneme); `--workers` ile paralellik ayarlanır. |
| Türkçe kalitesi | Gemini Türkçeyi iyi destekler. 50 yorumluk pilot kalitesini ölçer (aşağıda). Pilot zayıf çıkarsa daha güçlü model: `gemini-3.8-flash` (girdi $0,75, çıktı $3,75; 1 Ocak 2027'de iki katına çıkıyor), tahmini maliyet ≈ 5 USD. |
| Determinizm | `temperature=0`, JSON modu (`response_format=json_object`). |

Fiyatlar: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing), Ekim 2026.

## Kurulum

`.env` (git'e girmez, bkz. `.env.example`):

```bash
LLM_API_KEY=<Gemini API anahtarı (AI Studio, faturalandırma açık proje)>
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
LLM_REASONING_EFFORT=low   # düşünme token'larını azaltır (çıktı olarak faturalanır); Gemini "none" kabul etmiyor
```

Model `temperature` ya da `reasoning_effort` kabul etmezse: `LLM_TEMPERATURE=none`, `LLM_REASONING_EFFORT=` (boş).

## Çalıştırma

```bash
# 1) Pilot: örnekten sabit seed ile rastgele 50 yorum
uv run python -m absa.labeling.run --limit 50 --out data/interim/prelabels_pilot.csv \
    --price-in 0.30 --price-out 2.50 --price-cached 0.03

# 2) Tam çalıştırma (pilotun 50 yorumu önbellekten gelir)
uv run python -m absa.labeling.run --out data/interim/prelabels.csv \
    --price-in 0.30 --price-out 2.50 --price-cached 0.03
```

| Çıktı | İçerik |
|---|---|
| `prelabels.csv` | `review_id, text` ve 7 aspect etiketi. Yalnızca `schema.py`'ye göre geçerli satırlar. |
| `prelabels_failed.csv` | Geçersiz cevap (`invalid`) ya da API hatası (`error`) olan yorumlar, ham cevapla birlikte. |
| `prelabels_stats.json` / `.md` | Yorum sayısı, başarısız cevaplar, aspect başına etiket dağılımı, token ve toplam maliyet. Issue raporu için `.md` kopyalanır. |
| `prelabels_cache.jsonl` | Önbellek. Her sonuç geldiği anda diske yazılır. |

Hepsi `data/interim/` altındadır, git'e girmez ve Drive'da paylaşılır (bkz. `data/README.md`).

## Doğrulama ve yeniden deneme

- Her cevap `src/absa/labeling/validate.py` ile kontrol edilir: geçerli JSON, 7 aspect'in hepsi, başka anahtar yok, etiketler yalnızca `yok / olumlu / olumsuz / notr`.
- Geçersiz cevapta modele hatası söylenip düzeltilmiş cevap istenir (en fazla 3 deneme, `--max-attempts`). Yine geçersizse yorum `invalid` olarak işaretlenir ve `_failed.csv`'ye yazılır.
- API hatası olan yorumlar (`error`) bir sonraki çalıştırmada otomatik tekrar denenir. `invalid` olanları tekrar denemek için `--retry-invalid`.

## Önbellek ve kaldığı yerden devam

Komut yarıda kesilirse (çökme, Ctrl+C, kota) **aynı komutu tekrar çalıştırmak yeterlidir**: önbellekteki yorumlar atlanır.
Önbellek kayıtları model ve prompt sürümüyle birlikte tutulur; model ya da prompt değişirse sonuçlar karışmaz, yeni sürüm için yorumlar baştan etiketlenir.

## 50 yorumluk kontrol

1. Pilotu çalıştırın (yukarıda).
2. `prelabels_pilot.csv`'yi kopyalayın (`pilot_checked.csv`) ve her yorumu `docs/annotation/` kurallarına göre elle kontrol edip yanlış etiketleri düzeltin.
3. Karşılaştırın:

```bash
   uv run python -m absa.labeling.compare \
       --pred data/interim/prelabels_pilot.csv --gold data/interim/pilot_checked.csv
```

   Aspect başına doğruluk, kappa, kaçırılan (`missed`), fazladan işaretlenen (`extra`) ve duygusu yanlış (`wrong sent.`) etiketler yazdırılır; uyuşmazlıklar `prelabels_pilot_diff.csv`'ye yazılır.
4. Hataların ortak örüntüsüne göre prompt'u iyileştirin: `prompts/v1.md` dosyasını `v2.md` olarak kopyalayıp düzenleyin, `prompt.py` içinde `PROMPT_VERSION = "v2"` yapın, pilotu tekrar çalıştırıp karşılaştırın.
5. Sonuçları (her iki sürümün tablosu ve yapılan değişiklikler) issue #31'e yorum olarak ekleyin.

## Prompt

Prompt `docs/annotation/aspects.md` (aspect tanımları) ve `docs/annotation/boundary_rules.md` (sınır durumları, etiket kuralları) temel alınarak yazıldı:
rol, 4 etiket, 7 aspect tanımı, 10 sınır kuralı, etiket kuralları (`yok` / `notr` ayrımı, karışık duygu, olumsuzlama, ironi, emoji, karşılaştırma, tavsiye), JSON çıktı formatı ve 6 örnek.
Prompt'taki örneklerin şemaya uyduğu testle kontrol edilir.

## Sınırlar

- LLM etiketleri **öneridir**, altın etiket değildir. Etiketleyiciler her öneriyi kontrol eder.
- Ön etiket, etiketleyicileri önerilen etikete yönlendirebilir (anchoring). Bu yüzden kappa seti ön etiketsiz etiketlenir ve LLM ile insan uyumu ayrıca ölçülür.
- Maliyet, sağlayıcının döndürdüğü token sayılarından ve komut satırında verilen fiyatlardan hesaplanır; faturayla küçük farklar olabilir.
