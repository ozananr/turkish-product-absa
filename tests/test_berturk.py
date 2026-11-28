"""BERTurk eğitim kodu testleri: internet ve GPU gerekmez (küçük, rastgele bir BERT)."""

import json

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from absa.models.berturk import (  # noqa: E402
    AspectModel,
    TrainConfig,
    aspect_loss,
    class_weights,
    encode_labels,
    load_model,
    main,
    predict_labels,
    save_model,
    summarize,
    train,
)
from absa.schema import ASPECTS, LABELS  # noqa: E402

WORDS = ["kaliteli", "bozuldu", "kargo", "hızlı", "geç", "ucuz", "pahalı", "güzel", "ürün", "çok"]


def tiny_factory(model_name: str = "tiny", vocab_dir=None):
    """Küçük, rastgele ağırlıklı BERT + kelime listesinden tokenizer (indirme yok)."""
    import tempfile
    from pathlib import Path

    folder = Path(vocab_dir or tempfile.mkdtemp())
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (folder / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(folder / "vocab.txt"), do_lower_case=False
    )
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return transformers.BertModel(config), tokenizer


def _labels(n=40):
    rows = []
    for i in range(n):
        labels = {a: "yok" for a in ASPECTS}
        if i % 2 == 0:
            labels["kalite"], text = "olumlu", "çok kaliteli ürün"
        else:
            labels["kalite"], text = "olumsuz", "ürün bozuldu"
        if i % 3 == 0:
            labels["kargo_paketleme"], text = "olumlu", text + " kargo hızlı"
        rows.append({"review_id": f"r{i}", "text": text, "annotator": "x"} | labels)
    return pd.DataFrame(rows)


def _parts():
    df = _labels()
    return {
        "train": df.iloc[:30].reset_index(drop=True),
        "val": df.iloc[30:].reset_index(drop=True),
    }


def test_model_cikti_boyutu():
    encoder, tokenizer = tiny_factory()
    model = AspectModel(encoder)
    enc = tokenizer(["çok kaliteli", "kargo geç"], padding=True, return_tensors="pt")
    logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (2, len(ASPECTS), len(LABELS))


def test_etiket_kodlama_ve_sinif_agirliklari():
    df = _labels()
    y = encode_labels(df)
    assert y.shape == (len(df), len(ASPECTS))
    assert LABELS[int(y[0, ASPECTS.index("kalite")])] == "olumlu"
    w = class_weights(df)
    assert w.shape == (len(ASPECTS), len(LABELS))
    assert w[ASPECTS.index("fiyat"), LABELS.index("olumlu")] == 0  # hiç örneği yok
    kalite = ASPECTS.index("kalite")
    assert w[kalite, LABELS.index("olumlu")] > 0


def test_kayip_agirlikli_ve_agirliksiz():
    logits = torch.randn(4, len(ASPECTS), len(LABELS))
    targets = torch.zeros(4, len(ASPECTS), dtype=torch.long)
    plain = aspect_loss(logits, targets)
    weighted = aspect_loss(logits, targets, torch.ones(len(ASPECTS), len(LABELS)))
    assert torch.isclose(plain, weighted)


def test_egitim_val_ile_en_iyi_epoch():
    cfg = TrainConfig(lr=1e-3, epochs=3, batch_size=8, patience=5, class_weights=True)
    model, tokenizer, info = train(_parts(), cfg, factory=tiny_factory, log=lambda _: None)
    assert len(info["history"]) == 3
    assert info["val_macro_f1"] == max(h["val_macro_f1"] for h in info["history"])
    pred = predict_labels(model, tokenizer, ["çok kaliteli ürün"])
    assert set(pred) == set(ASPECTS) and pred["kalite"][0] in LABELS


def test_kaydet_ve_yukle_ayni_tahmin(tmp_path):
    cfg = TrainConfig(lr=1e-3, epochs=1, batch_size=8)
    model, tokenizer, info = train(_parts(), cfg, factory=tiny_factory, log=lambda _: None)
    save_model(model, tokenizer, tmp_path / "best", info)
    loaded, tok, meta = load_model(tmp_path / "best")
    texts = ["çok kaliteli ürün", "ürün bozuldu kargo geç"]
    assert predict_labels(loaded, tok, texts) == predict_labels(model, tokenizer, texts)
    assert meta["aspects"] == ASPECTS and meta["max_length"] == 128


def test_ozet_tablosu_tohum_ortalamasi():
    runs = [
        {"config": {"lr": 3e-5, "class_weights": True, "seed": s}, "epoch": 2, "val_macro_f1": f}
        for s, f in [(42, 0.6), (43, 0.62)]
    ]
    table = summarize(runs)
    assert "| 3e-05 | on | 43 | 2 | 0.620 |" in table and "0.610 ± 0.010" in table


def test_cli_tarama(tmp_path):
    df = _labels()
    labels, split, out = tmp_path / "labels.csv", tmp_path / "split.csv", tmp_path / "out"
    df.to_csv(labels, index=False)
    pd.DataFrame(
        {"review_id": df["review_id"], "split": ["train"] * 30 + ["val"] * 6 + ["test"] * 4}
    ).to_csv(split, index=False)
    runs = main(
        [
            *("--labels", str(labels), "--split", str(split), "--out", str(out)),
            *("--lr", "1e-3", "5e-4", "--class-weights", "off", "on", "--epochs", "1"),
        ],
        factory=tiny_factory,
    )
    assert len(runs) == 4
    assert (out / "best" / "heads.pt").exists() and (out / "summary.md").exists()
    saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(saved) == 4
