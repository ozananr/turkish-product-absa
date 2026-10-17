import sys
from pathlib import Path

# `python app.py` ile paket kurulu olmasa da src/ altından import edebilmek için.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from absa.inference import predict  # noqa: E402
from absa.schema import ASPECT_DISPLAY, LABEL_DISPLAY, NO_MENTION  # noqa: E402

EXAMPLES = [
    ["Ürün çok kaliteli ama kargo çok geç geldi."],
    ["Fiyatına göre gayet iyi, görüntüsü de resimdeki gibi."],
    ["İki günde bozuldu, satıcı da iade etmedi."],
    ["Bir beden büyük geldi ama kumaşı güzel."],
]

HEADERS = ["Özellik", "Duygu", "Güven"]


def analyze(text: str):
    try:
        result = predict(text or "")
    except (ValueError, TypeError):
        gr.Warning("Lütfen analiz etmek için bir yorum yazın.")
        return pd.DataFrame(columns=HEADERS), ""

    # "yok" olanlar tabloda gösterilmez; kapalı bir bölümde ayrıca listelenir (FR-4).
    rows = [
        [
            ASPECT_DISPLAY[aspect],
            LABEL_DISPLAY[r["label"]],
            f"{r['confidence']:.0%}",
        ]
        for aspect, r in result.items()
        if r["label"] != NO_MENTION
    ]
    hidden = [ASPECT_DISPLAY[a] for a, r in result.items() if r["label"] == NO_MENTION]
    hidden_text = ", ".join(hidden) if hidden else "Yok"

    return pd.DataFrame(rows, columns=HEADERS), hidden_text


with gr.Blocks(title="Özellik Bazlı Duygu Analizi") as demo:
    gr.Markdown(
        "# Türkçe Ürün Yorumlarında Özellik Bazlı Duygu Analizi\n"
        "_Şu an **sahte model** çalışıyor; sonuçlar gerçek tahmin değildir._"
    )
    review = gr.Textbox(label="Yorum", lines=4, placeholder="Ürün yorumunu buraya yazın...")
    button = gr.Button("Analiz et", variant="primary")
    table = gr.Dataframe(headers=HEADERS, interactive=False, label="Sonuç")
    with gr.Accordion("Bahsedilmeyen özellikler (yok)", open=False):
        hidden_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=review)

    button.click(analyze, inputs=review, outputs=[table, hidden_box])
    review.submit(analyze, inputs=review, outputs=[table, hidden_box])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
