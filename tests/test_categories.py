import pytest

from absa.data.categories import OTHER, UNKNOWN, assign_category


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Kadın Penye Likralı Konforlu Uzun Tayt TYC00236233266", "giyim"),
        ("Ay Ve Güneş Baskılı Sweat Unısex Oversıze Sweatshırt", "giyim"),
        ("%100 Doğal Roll On Deodorant Kadınlar İçin", "kozmetik_bakim"),
        ("Aqua Thermal Kepek Karşıtı Şampuan 300 ml", "kozmetik_bakim"),
        ("C23 Kadın Parfüm EDP 50 ml", "kozmetik_bakim"),
        ("Emici Külot Lohusa Pedi Kadın L Beden", "hijyen_ped"),  # ped, külot'tan önce
        ("Platinum Hijyenik Ped Gündüz Gece 64 Adet", "hijyen_ped"),
        ("Deniz ve Sörf Ayakkabısı fa652012", "ayakkabi"),
        ("Kadın Kalın Taban Süet Bot Kışlık Bot", "ayakkabi"),
        ("Yeni Sezon Kadın Güneş Gözlüğü ML458", "aksesuar"),
        ("Gnd Siyah Anne Bebek Bakım Sırt Çantası", "aksesuar"),  # aksesuar, bakım'dan önce
        ("Reflex Yetişkin Kaleci Eldiveni", "spor"),
        ("Fitness Bilekliği Ağırlık Bilekliği", "spor"),
        ("Çıtçıtlı Bone Hazır Bone Tesettür", "giyim"),
        ("Kalıcı Doğal Renkler Saç Boyası 8-0", "kozmetik_bakim"),
        ("Doğal Roll-on Cherish 60 ml", "kozmetik_bakim"),
        ("Kadın Gümüş Küçük Boy Sallantılı Fiyonk Küpe", "aksesuar"),
        ("Bilinmeyen Bir Şey 123", OTHER),
    ],
)
def test_kategori_atama(name, expected):
    assert assign_category(name, "trendyol") == expected


def test_pedal_ped_sayilmaz():
    assert assign_category("Bisiklet Pedalı", "trendyol") == OTHER


def test_urun_adi_yoksa():
    assert assign_category(None, "trsa") == UNKNOWN
    assert assign_category(float("nan"), "trsa") == UNKNOWN
    assert assign_category("", "trendyol") == OTHER
