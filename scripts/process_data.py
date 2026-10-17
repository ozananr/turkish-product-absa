"""Eski dummy betiğin yerine: gerçek veri hattını çalıştırır.

Kullanım:
    uv run python scripts/process_data.py --trsa <TRSAv1.csv> --trendyol <trendyol.csv>
"""

from absa.data.pipeline import main

if __name__ == "__main__":
    main()
