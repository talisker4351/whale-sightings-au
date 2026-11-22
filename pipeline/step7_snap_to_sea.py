"""v2: 地図の印が陸の上に出ないように、海の上へ寄せる(step6 のあとに実行)。

step6 は、マス目の中の記録の「平均の位置」に印を置く。記録が岬の両側に分かれていたり、
展望台など陸から見た人の位置で記録されていたりすると、平均が陸の上になることがある。
ここでは、陸の形(Natural Earth 10m の陸地データ)を使って、陸の上にある印だけを
いちばん近い海岸の少し沖(約1.5km)へ動かす。海の上にある印は動かさない。

使い方:
    python pipeline/step7_snap_to_sea.py

必要なもの: numpy, pandas, matplotlib, requests(陸地データを初回だけダウンロードする)

入力・出力:
    data/cell_points.csv を上書きする(列は glat, glon, plat, plon のまま)
    data/raw/ne_10m_land.geojson  陸地データ(初回に自動でダウンロード。約10MB)
"""
import json
import sys
from pathlib import Path as FilePath

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path

sys.path.insert(0, str(FilePath(__file__).resolve().parents[1]))
from config import DATA, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW

LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_10m_land.geojson")
LAND_FILE = RAW / "ne_10m_land.geojson"
OFFSHORE_KM = 1.5     # 海岸から、この距離だけ沖に置く
MARGIN_DEG = 0.01     # 置いた点の上下左右(約1km)も海であることを確かめる
SEARCH_DEG = 0.6      # 海岸を探す範囲(度)。約60km


def load_land() -> list:
    """オーストラリア周辺の陸地の輪郭を、点の内外判定ができる形で読み込む。"""
    if not LAND_FILE.exists():
        print("陸地データをダウンロードします:", LAND_URL)
        r = requests.get(LAND_URL, timeout=120)
        r.raise_for_status()
        LAND_FILE.write_bytes(r.content)
    g = json.loads(LAND_FILE.read_text())
    rings = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.array(poly[0])            # 外側の輪郭だけ使う(湖は陸として扱う)
            if (ring[:, 0].max() < LON_MIN - 1 or ring[:, 0].min() > LON_MAX + 1
                    or ring[:, 1].max() < LAT_MIN - 1 or ring[:, 1].min() > LAT_MAX + 1):
                continue
            rings.append(ring)
    return rings


def on_land(rings, lon, lat) -> np.ndarray:
    """点(経度, 緯度)が陸の上なら True。"""
    pts = np.c_[np.atleast_1d(lon), np.atleast_1d(lat)]
    inside = np.zeros(len(pts), bool)
    for ring in rings:
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        near = ((pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0])
                & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        if near.any():
            inside[near] |= Path(ring).contains_points(pts[near])
    return inside


def clearly_sea(rings, lon, lat) -> bool:
    """点と、その上下左右(約1km)がすべて海か。細い川や入り江に置かないため。"""
    d = MARGIN_DEG
    lons = [lon, lon + d, lon - d, lon, lon]
    lats = [lat, lat, lat, lat + d, lat - d]
    return not on_land(rings, lons, lats).any()


def snap(rings, coast, lon, lat):
    """陸の上の点を、近い海岸から少し沖へ動かした位置を返す。見つからなければ元の位置。"""
    k = np.cos(np.radians(lat))                       # 経度1度の長さは緯度で変わる
    near = coast[(abs(coast[:, 0] - lon) < SEARCH_DEG) & (abs(coast[:, 1] - lat) < SEARCH_DEG)]
    if len(near) == 0:
        return lon, lat
    dist = np.hypot((near[:, 0] - lon) * k, near[:, 1] - lat)
    for i in np.argsort(dist)[:300]:                  # 近い海岸の点から順に試す
        cx, cy = near[i]
        dx, dy = (cx - lon) * k, cy - lat
        length = np.hypot(dx, dy) or 1e-9
        ux, uy = dx / length, dy / length             # 陸の点 → 海岸 の向き
        for km in (OFFSHORE_KM, OFFSHORE_KM * 2, OFFSHORE_KM * 4):
            step = km / 111.0
            nx, ny = cx + ux * step / k, cy + uy * step
            if clearly_sea(rings, nx, ny):
                return round(nx, 4), round(ny, 4)
    return lon, lat


def main() -> None:
    pts = pd.read_csv(DATA / "cell_points.csv")
    rings = load_land()
    coast = np.vstack(rings)
    land = on_land(rings, pts["plon"].values, pts["plat"].values)
    print("印の数", len(pts), "/ 陸の上", int(land.sum()))

    moved = []
    for i in np.where(land)[0]:
        lon0, lat0 = pts.at[i, "plon"], pts.at[i, "plat"]
        lon1, lat1 = snap(rings, coast, lon0, lat0)
        km = float(np.hypot((lon1 - lon0) * 111 * np.cos(np.radians(lat0)), (lat1 - lat0) * 111))
        pts.at[i, "plon"], pts.at[i, "plat"] = lon1, lat1
        moved.append(km)

    still = on_land(rings, pts["plon"].values, pts["plat"].values)
    if moved:
        print("動かした距離(km): 中央値", round(float(np.median(moved)), 1),
              "/ 最大", round(max(moved), 1))
    print("まだ陸の上", int(still.sum()))
    pts.to_csv(DATA / "cell_points.csv", index=False)
    print("保存:", DATA / "cell_points.csv")


if __name__ == "__main__":
    main()
