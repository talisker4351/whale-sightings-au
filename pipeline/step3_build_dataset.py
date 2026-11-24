"""v1a: 「いた(1)・いなかった(0)」の学習用データを作る。

考え方(ここが最も大事な設計判断):
    使う年は2000年〜昨年(いちばん新しい年は途中なので除く)。
    記録は「いた」しかないので、「いなかった」を自分で決める必要がある。
    ここでは、一度でも記録があるマスを「人が見ている場所」とみなし、
    そのマスの各年・各月に記録があれば1、なければ0とする。
    人の多い場所への偏りは完全には取り除けない。改善案は HANDOFF_ja.md を参照。

使い方:
    python pipeline/step3_build_dataset.py

出力:
    data/dataset.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, FEATURES, YEAR_MIN
from features import add_features  # noqa: E402  (同じフォルダ)


def main() -> None:
    s = pd.read_csv(DATA / "sightings.csv")
    # いちばん新しい年は途中(まだ来ていない月も「記録なし=0」になる)なので使わない。
    # 2000年〜昨年の、全部そろった年だけで学習・評価する(2026-11-24 修正)
    last_full = int(s["year"].max()) - 1
    s = s[s["year"].between(YEAR_MIN, last_full)]
    print("使う年", YEAR_MIN, "〜", last_full)
    sst = pd.read_csv(DATA / "sst_cell_month.csv")

    seen = (s.groupby(["glat", "glon", "year", "month"])
              .size()
              .reset_index(name="n"))
    cells = seen[["glat", "glon"]].drop_duplicates()
    years = pd.DataFrame({"year": sorted(s["year"].unique())})
    months = pd.DataFrame({"month": range(1, 13)})

    full = cells.merge(years, how="cross").merge(months, how="cross")
    full = full.merge(seen, on=["glat", "glon", "year", "month"], how="left")
    full["y"] = (full["n"].fillna(0) > 0).astype(int)

    full = full.merge(sst, on=["glat", "glon", "year", "month"], how="left")
    n0 = len(full)
    full = full.dropna(subset=["sst"])
    print("水温がない行を除く", n0, "->", len(full))

    full = add_features(full)
    print("行数", len(full), "/ 1の割合", round(full["y"].mean(), 4))
    full[["glat", "glon", "year", "month", "y"] + [c for c in FEATURES if c not in ("glat", "glon")]].to_csv(
        DATA / "dataset.csv", index=False)
    print("保存:", DATA / "dataset.csv")


if __name__ == "__main__":
    main()
