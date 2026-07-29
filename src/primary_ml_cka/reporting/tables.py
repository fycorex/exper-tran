from pathlib import Path

import pandas as pd


def csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="all_results", index=False)
