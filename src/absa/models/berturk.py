"""BERTurk: ortak kodlayıcı + 7 aspect başlığı (issue #65).

Model: dbmdz/bert-base-turkish-cased kodlayıcısı; [CLS] temsili her aspect için ayrı bir doğrusal
başlığa (4 sınıf: yok, olumlu, olumsuz, notr) verilir. Kayıp, 7 başlığın çapraz entropi
kayıplarının ortalamasıdır. İsteğe bağlı olarak her aspect için sınıf ağırlıklı kayıp kullanılır.

Eğitim `split_v1` train parçasında yapılır; her epoch sonunda val macro-F1 ölçülür, en iyi epoch
saklanır ve iyileşme durunca eğitim erken durdurulur. Test parçası burada KULLANILMAZ (#66).

Kullanım (GPU önerilir, ör. Google Colab):
    # Tek ayar
    python -m absa.models.berturk \\
        --labels data/interim/labeling/final_labels.csv \\
        --split data/interim/splits/split_v1.csv \\
        --lr 3e-5 --class-weights on

    # Tarama: her lr x class-weights x seed birleşimi; en iyi val modeli out/best/ altına
    python -m absa.models.berturk --labels ... --split ... \\
        --lr 2e-5 3e-5 5e-5 --class-weights off on --seeds 42

Çıktılar (--out, varsayılan outputs/berturk/; git'e girmez):
    summary.md / summary.json   çalıştırmaların val macro-F1'i (+ aynı ayarın tohum ort. ± std)
    best/          en iyi val modeli: kodlayıcı + tokenizer + heads.pt + absa_config.json
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from absa.models.baseline_full import load_split_data
from absa.models.evaluate import evaluate
from absa.schema import ASPECTS, LABELS

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 128
HEADS_FILE = "heads.pt"
CONFIG_FILE = "absa_config.json"


@dataclass
class TrainConfig:
    model_name: str = MODEL_NAME
    lr: float = 3e-5
    batch_size: int = 16
    epochs: int = 5
    patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    class_weights: bool = False
    max_length: int = MAX_LENGTH
    seed: int = 42


class AspectModel(nn.Module):
    """Ortak kodlayıcı + aspect başına bir doğrusal başlık. Çıktı: [batch, n_aspects, n_labels]."""

    def __init__(
        self,
        encoder: nn.Module,
        n_aspects: int = len(ASPECTS),
        n_labels: int = len(LABELS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(hidden, n_labels) for _ in range(n_aspects))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return torch.stack([head(cls) for head in self.heads], dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_labels(df: pd.DataFrame) -> torch.Tensor:
    """[n_reviews, n_aspects] etiket indeksleri (LABELS sırası)."""
    index = {label: i for i, label in enumerate(LABELS)}
    return torch.tensor([[index[v] for v in df[a]] for a in ASPECTS], dtype=torch.long).T


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """[n_aspects, n_labels] ters sınıf frekansı ağırlıkları: n / (k * sayı). Hiç örneği olmayan
    sınıfın ağırlığı 0'dır (hedefte hiç geçmez)."""
    n, k = len(df), len(LABELS)
    rows = []
    for a in ASPECTS:
        counts = df[a].value_counts()
        rows.append([n / (k * counts[lab]) if counts.get(lab, 0) else 0.0 for lab in LABELS])
    return torch.tensor(rows, dtype=torch.float)


def aspect_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """7 aspect'in çapraz entropi kayıplarının ortalaması."""
    losses = [
        nn.functional.cross_entropy(
            logits[:, i], targets[:, i], weight=None if weights is None else weights[i]
        )
        for i in range(logits.shape[1])
    ]
    return torch.stack(losses).mean()


def _batches(n: int, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def predict_proba(
    model: AspectModel,
    tokenizer,
    texts: Sequence[str],
    max_length: int = MAX_LENGTH,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """[n_texts, n_aspects, n_labels] olasılıklar."""
    device = device or next(model.parameters()).device
    model.eval()
    texts = list(texts)
    out = []
    with torch.no_grad():
        for idx in _batches(len(texts), batch_size, shuffle=False):
            enc = tokenizer(
                [texts[i] for i in idx],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            out.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0, len(ASPECTS), len(LABELS))


def predict_labels(model, tokenizer, texts, **kwargs) -> dict[str, list[str]]:
    """evaluate() ile aynı biçim: {aspect: [label, ...]}."""
    ids = predict_proba(model, tokenizer, texts, **kwargs).argmax(dim=-1)
    return {a: [LABELS[int(i)] for i in ids[:, j]] for j, a in enumerate(ASPECTS)}


def default_factory(model_name: str):
    return AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)


def train(
    parts: dict[str, pd.DataFrame],
    cfg: TrainConfig,
    factory: Callable = default_factory,
    device: torch.device | None = None,
    log: Callable[[str], None] = print,
) -> tuple[AspectModel, object, dict]:
    """Train'de eğitir, her epoch val macro-F1 ölçer; en iyi epoch'un modelini döndürür."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    encoder, tokenizer = factory(cfg.model_name)
    model = AspectModel(encoder).to(device)

    train_df, val_df = parts["train"], parts["val"]
    texts = train_df["text"].tolist()
    targets = encode_labels(train_df)
    weights = class_weights(train_df).to(device) if cfg.class_weights else None
    y_val = {a: val_df[a].tolist() for a in ASPECTS}

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    steps = cfg.epochs * ((len(texts) + cfg.batch_size - 1) // cfg.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(cfg.seed)

    best = {"val_macro_f1": -1.0, "epoch": 0}
    best_state, history, waited = None, [], 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start, total = time.time(), 0.0
        for idx in _batches(len(texts), cfg.batch_size, shuffle=True, generator=generator):
            enc = tokenizer(
                [texts[i] for i in idx],
                truncation=True,
                max_length=cfg.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(enc["input_ids"], enc["attention_mask"])
            loss = aspect_loss(logits.float(), targets[idx].to(device), weights)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(idx)

        pred = predict_labels(model, tokenizer, val_df["text"], max_length=cfg.max_length)
        val_f1 = evaluate(y_val, pred)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / len(texts), "val_macro_f1": val_f1})
        log(
            f"  epoch {epoch}: loss {total / len(texts):.4f}  val macro-F1 {val_f1:.3f}"
            f"  ({time.time() - start:.0f}s)"
        )
        if val_f1 > best["val_macro_f1"]:
            best = {"val_macro_f1": val_f1, "epoch": epoch}
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            waited = 0
        else:
            waited += 1
            if waited >= cfg.patience:
                log(f"  erken durdurma (epoch {epoch})")
                break

    model.load_state_dict(best_state)
    return model, tokenizer, {"config": asdict(cfg), "history": history, **best}


def save_model(model: AspectModel, tokenizer, out: Path, info: dict) -> None:
    """Kodlayıcı + tokenizer (Hugging Face biçimi), başlıklar ve etiket şeması."""
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(model.heads.state_dict(), out / HEADS_FILE)
    meta = {"aspects": ASPECTS, "labels": LABELS, "max_length": info["config"]["max_length"]}
    (out / CONFIG_FILE).write_text(
        json.dumps(meta | {"training": info}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_model(path: Path, device: torch.device | str = "cpu") -> tuple[AspectModel, object, dict]:
    """save_model ile kaydedilen modeli yükler (#67 predict() bunu kullanacak)."""
    path = Path(path)
    meta = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
    if meta["aspects"] != ASPECTS or meta["labels"] != LABELS:
        raise ValueError("Kaydedilen modelin aspect/etiket şeması schema.py ile uyuşmuyor")
    model = AspectModel(AutoModel.from_pretrained(path))
    model.heads.load_state_dict(torch.load(path / HEADS_FILE, map_location="cpu"))
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(path), meta


def summarize(runs: list[dict]) -> str:
    lines = [
        "| lr | class weights | seed | best epoch | val macro-F1 |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in sorted(runs, key=lambda r: -r["val_macro_f1"]):
        c = r["config"]
        lines.append(
            f"| {c['lr']:g} | {'on' if c['class_weights'] else 'off'} | {c['seed']} "
            f"| {r['epoch']} | {r['val_macro_f1']:.3f} |"
        )
    groups: dict[tuple, list[float]] = {}
    for r in runs:
        key = (r["config"]["lr"], r["config"]["class_weights"])
        groups.setdefault(key, []).append(r["val_macro_f1"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        lines += [
            "",
            "| lr | class weights | seeds | val macro-F1 mean ± std |",
            "|---:|---|---:|---|",
        ]
        for (lr, cw), vals in multi.items():
            lines.append(
                f"| {lr:g} | {'on' if cw else 'off'} | {len(vals)} "
                f"| {fmean(vals):.3f} ± {pstdev(vals):.3f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None, factory: Callable = default_factory) -> list[dict]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True, help="final_labels.csv")
    parser.add_argument("--split", type=Path, required=True, help="split_v1.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs/berturk"))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5])
    parser.add_argument("--class-weights", choices=["on", "off"], nargs="+", default=["off"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args(argv)

    parts = load_split_data(args.labels, args.split)
    print(f"train {len(parts['train'])} / val {len(parts['val'])} (test bu adımda kullanılmaz)")
    runs, best_f1 = [], -1.0
    for lr, cw, seed in itertools.product(args.lr, args.class_weights, args.seeds):
        cfg = TrainConfig(
            model_name=args.model_name,
            lr=lr,
            class_weights=cw == "on",
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        print(f"\n=== lr={lr:g} class_weights={cw} seed={seed} ===")
        model, tokenizer, info = train(parts, cfg, factory=factory)
        runs.append(info)
        if info["val_macro_f1"] > best_f1:
            best_f1 = info["val_macro_f1"]
            save_model(model, tokenizer, args.out / "best", info)
            print(f"  yeni en iyi model kaydedildi: {args.out / 'best'}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = summarize(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print("\n" + table)
    print(f"\nYazıldı: {args.out}/ (summary.md, summary.json, best/)")
    return runs


if __name__ == "__main__":
    main()
