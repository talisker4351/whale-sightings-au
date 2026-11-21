"""v0: ALAの目撃記録を洗って、マス目×月の件数表を作る。

使い方:
    python pipeline/step1_prepare_sightings.py data/raw/records.zip
    (CSVを直接渡してもよい)

出力:
    data/sightings.csv   洗ったあとの記録(1行1件)
    data/grid_month.csv  マス目×月の件数(v0のアプリが読む)
"""
import glob
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW, SPECIES, STEP


def find_csv(src: str) -> str:
    """ZIPなら展開し、いちばん大きいCSVを記録の本体とみなす。"""
    if src.endswith(".zip"):
        out = RAW / "unzipped"
        zipfile.ZipFile(src).extractall(out)
        csvs = glob.glob(str(out / "**" / "*.csv"), recursive=True)
        return max(csvs, key=os.path.getsize)
    return src


def main(src: str) -> None:
    path = find_csv(src)
    print("使うファイル:", path)

    # 1. 読み込む(必要な4列だけ)
    cols = ["decimalLatitude", "decimalLongitude", "eventDate", "scientificName"]
    df = pd.read_csv(path, usecols=cols)
    df = df.rename(columns={"decimalLatitude": "lat",
                            "decimalLongitude": "lon",
                            "eventDate": "date"})
    print("読み込み", len(df))

    # 2. 種で絞る
    df = df[df["scientificName"].str.contains(SPECIES, na=False)]
    print("種で絞る", len(df))

    # 3. 洗う(欠損、範囲外、重複)
    df = df.dropna(subset=["lat", "lon", "date"])
    print("欠損を除く", len(df))
    df = df[df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)]
    print("範囲内", len(df))
    df = df.drop_duplicates(subset=["lat", "lon", "date"])
    print("重複を除く", len(df))

    # 4. 日付から年と月を作る
    df["date"] = df["date"].astype(str)
    df = df[df["date"].str.len() >= 10].copy()          # 年だけの日付を除く
    # 1月1日は「日付が分からず仮に入れた値」の可能性が高い(1月64件中27件が1月1日だった)ので除く
    df = df[df["date"].str[5:10] != "01-01"].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce",
                                format="ISO8601", utc=True)
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    print("日付が読めた", len(df))

    # 5. マス目に丸めて数える
    df["glat"] = (df["lat"] / STEP).round() * STEP
    df["glon"] = (df["lon"] / STEP).round() * STEP
    grid = (df.groupby(["glat", "glon", "month"])
              .size()
              .reset_index(name="n"))
    assert grid["n"].sum() == len(df), "検算が合わない"
    print("マス目×月", len(grid), "/ 検算OK")

    df[["lat", "lon", "date", "year", "month", "glat", "glon"]].to_csv(
        DATA / "sightings.csv", index=False)
    grid.to_csv(DATA / "grid_month.csv", index=False)
    print("保存:", DATA / "sightings.csv", DATA / "grid_month.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RAW / "records.zip"))
