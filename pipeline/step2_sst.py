"""v0.5: 海面水温(NOAA OISST v2 高解像度、月平均)を、マス目×年×月に付ける。

使い方:
    python pipeline/step2_sst.py

必要なもの: xarray と netCDF4
ネット経由で、オーストラリア周辺の範囲だけを、1年分ずつ読み込む(10分前後かかることがある)。
受信が途中で切れると値が0になることがあるため、年ごとに値を確かめ、おかしければやり直す。

出力:
    data/sst_cell_month.csv  glat, glon, year, month, sst(学習に使う)
    data/sst_clim.csv        glat, glon, month, sst(年をならした平年値。予測に使う)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, SST_URL,
                    YEAR_MIN)


def looks_valid(v: np.ndarray) -> bool:
    """受け取った水温が正常か。海の水温がちょうど0度になることはまずないので、0があれば失敗とみなす。"""
    sea = v[~np.isnan(v)]
    return sea.size > 0 and np.count_nonzero(sea == 0) == 0 \
        and sea.min() > -3 and sea.max() < 40


def load_year(da: xr.DataArray, year: int, tries: int = 4) -> xr.DataArray:
    for attempt in range(1, tries + 1):
        try:
            part = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
            if looks_valid(part.values):
                return part
            print(f"  {year}年: 値が正しくないので、やり直します({attempt}回目)")
        except Exception as e:  # 通信の失敗など
            print(f"  {year}年: 受信に失敗。やり直します({attempt}回目): {e}")
        time.sleep(5 * attempt)
    raise RuntimeError(f"{year}年の水温が取得できませんでした。時間をおいて実行し直してください。")


def main() -> None:
    sightings = pd.read_csv(DATA / "sightings.csv")
    cells = sightings[["glat", "glon"]].drop_duplicates().reset_index(drop=True)
    print("マス目の数", len(cells))

    ds = xr.open_dataset(SST_URL)
    # 緯度が北→南の順のデータにも対応する
    lat_up = bool(ds["lat"][0] < ds["lat"][-1])
    lat_slice = slice(LAT_MIN - 1, LAT_MAX + 1) if lat_up else slice(LAT_MAX + 1, LAT_MIN - 1)
    # OISSTの経度は0〜360度。オーストラリアはそのままの値で切り出せる
    da = ds["sst"].sel(lat=lat_slice, lon=slice(LON_MIN - 1, LON_MAX + 1))

    years = sorted(y for y in set(pd.to_datetime(ds["time"].values).year) if y >= YEAR_MIN)
    parts = []
    for y in years:
        parts.append(load_year(da, y))
        print(f"  {y}年 OK")
    box = xr.concat(parts, dim="time")
    print("読み込んだ範囲", dict(box.sizes))

    # 海岸に近いマスは陸と判定されて空欄のことがあるので、周り3×3の平均で埋める
    around = box.rolling(lat=3, lon=3, center=True, min_periods=1).mean()
    box = box.fillna(around)

    # 各マス目に最も近い格子点の値を取り出す
    pts = box.sel(lat=xr.DataArray(cells["glat"].values, dims="cell"),
                  lon=xr.DataArray(cells["glon"].values, dims="cell"),
                  method="nearest")
    out = pts.to_dataframe(name="sst").reset_index()
    out["glat"] = cells["glat"].values[out["cell"].values]
    out["glon"] = cells["glon"].values[out["cell"].values]
    out["year"] = pd.to_datetime(out["time"]).dt.year
    out["month"] = pd.to_datetime(out["time"]).dt.month
    out = out[["glat", "glon", "year", "month", "sst"]]

    # 確認: 値の範囲と、シドニー沖の季節変化(2〜3月が暖かく、8月ごろが冷たいはず)
    print("水温が空欄の行", int(out["sst"].isna().sum()), "/", len(out))
    print("水温の最小・最大", round(out["sst"].min(), 1), round(out["sst"].max(), 1))
    syd = out[out["glat"].between(-35, -33) & (out["glon"] > 150)]
    print("シドニー沖の月平均", syd.groupby("month")["sst"].mean().round(1).to_dict())
    if not looks_valid(out["sst"].to_numpy()):
        raise RuntimeError("水温の値が正しくありません。保存せずに終了します。")

    clim = (out.groupby(["glat", "glon", "month"])["sst"]
               .mean()
               .reset_index())
    out.to_csv(DATA / "sst_cell_month.csv", index=False)
    clim.to_csv(DATA / "sst_clim.csv", index=False)
    print("保存:", DATA / "sst_cell_month.csv", DATA / "sst_clim.csv")


if __name__ == "__main__":
    main()
