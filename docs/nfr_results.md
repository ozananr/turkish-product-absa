# Non-functional requirements: results (#68)

Measured with the trained BERTurk model (`models/berturk_v1`, #65/#66) on CPU, with
`python -m absa.nfr` (`src/absa/nfr.py`). The command starts in a new process, so the load time
includes importing torch / transformers (cold start). After 3 warm-up calls, 100 reviews are
predicted one by one.

## Summary

| Requirement | Limit | Windows | Linux | Result |
|---|---|---|---|---|
| NFR-1 single prediction on CPU (95th percentile) | ≤ 2 s | 98 ms | 410 ms | pass |
| NFR-2 model loaded once at start, load time | ≤ 60 s | 8.0 s | 10.4 s | pass |
| NFR-4 user input not stored or logged | – | test passes | test passes (CI) | pass |
| NFR-6 runs on Windows and Linux | – | app runs | app runs | pass |

## NFR-1 and NFR-2: timing

| Run | Environment | Inputs | Load | Median | 95th pct. | Max | Truncated |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Windows 11, Intel 12th gen, Python 3.12.15, torch 2.15.0+cpu, 8 threads | built-in examples (short, medium, long) | 8.0 s | 20 ms | 98 ms | 107 ms | 9 / 100 |
| 2 | same | 100 real reviews from `final_labels.csv` | 8.1 s | 23 ms | 48 ms | 100 ms | 0 / 100 |
| 3 | Linux (Google Colab CPU runtime), Python 3.13, torch 2.14.1+cpu, 1 thread | built-in examples | 10.4 s | 92 ms | 410 ms | 624 ms | 9 / 100 |

- Even on a single CPU thread (run 3) the slowest prediction is below one third of the 2 s limit.
- The built-in examples include a review longer than 128 tokens, so the truncation path (FR-7) is
  part of the measurement.
- The model is loaded once per process and reused for every request; this is covered by
  `tests/test_inference.py::test_model_bir_kez_yuklenir`. The app loads it at start, so the first
  analysis does not wait.

Reproduce:

```bash
uv run python -m absa.nfr --n 100
uv run python -m absa.nfr --texts data/interim/labeling/final_labels.csv --n 100
```

## NFR-4: privacy

`tests/test_nfr.py::test_girdi_kaydedilmez_ve_loglanmaz` predicts a marked review and checks that
no file is created and that the text does not appear in log records, stdout or stderr. It runs in
CI (mock) and locally with the real model. The app keeps no history: Gradio flagging is not used
and `app.py` does not write the input anywhere. The training data is masked (#18 pipeline).

## NFR-6: Windows and Linux

- **Windows 11**, Python 3.12: `uv run python app.py` loads the model and serves the app on
  http://127.0.0.1:7860; predictions checked with the four example reviews (screenshots in #75 / #68).
- **Linux** (Google Colab CPU runtime): the timing command above, and the app module builds its
  interface and analyzes a review (`app.analyze("Ürün çok kaliteli ama kargo çok geç geldi.")` →
  Kalite olumlu 98 %, Kargo olumsuz 84 %).
- **CI** (GitHub Actions, Ubuntu, Python 3.12) runs the full test suite with the mock on every pull
  request.
