"""v2: 地図の印が陸の上や細い川・入り江に出ないように、開けた海の上へ寄せる(step6 のあとに実行)。

step6 は、マス目の中の記録の「平均の位置」に印を置く。記録が岬の両側に分かれていたり、
展望台など陸から見た人の位置で記録されていたりすると、平均が陸の上になることがある。
ここでは、陸の形(Natural Earth 10m の陸地データ)を約500m四方の細かいマス目の地図に描き、
「陸から約1.5km以上離れた海」を「開けた海」とする。開けた海にない印だけを、
いちばん近い開けた海へ動かす。すでに開けた海にある印は動かさない。

使い方:
    python pipeline/step7_snap_to_sea.py

必要なもの: numpy, pandas, Pillow(matplotlib と一緒に入っている), requests
            (陸地データを初回だけダウンロードする)

入力・出力:
    data/cell_points.csv を上書きする(列は glat, glon, plat, plon のまま)
    data/raw/ne_10m_land.geojson  陸地データ(初回に自動でダウンロード。約10MB)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, RAW

LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_10m_land.geojson")
LAND_FILE = RAW / "ne_10m_land.geojson"
RES = 0.005           # 細かいマス目の大きさ(度)。約500m
MARGIN_KM = 1.5       # 陸からこの距離より離れていれば「開けた海」
SEARCH_DEG = 0.6      # 開けた海を探す範囲(度)。約60km

# 細かい地図の範囲(調べる範囲より1度広く取る)
W0, W1 = LON_MIN - 1, LON_MAX + 1
S0, N0 = LAT_MIN - 1, LAT_MAX + 1


def load_rings() -> list:
    """オーストラリア周辺の陸地の輪郭(経度, 緯度の点の並び)を読み込む。"""
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
            if (ring[:, 0].max() < W0 or ring[:, 0].min() > W1
                    or ring[:, 1].max() < S0 or ring[:, 1].min() > N0):
                continue
            rings.append(ring)
    return rings


def to_px(lon, lat):
    """経度・緯度 → 細かい地図の列・行の番号。"""
    col = np.floor((np.asarray(lon) - W0) / RES).astype(int)
    row = np.floor((N0 - np.asarray(lat)) / RES).astype(int)
    return col, row


def land_raster(rings) -> np.ndarray:
    """陸=True、海=False の細かい地図(行=北から南、列=西から東)を作る。"""
    w = int(round((W1 - W0) / RES))
    h = int(round((N0 - S0) / RES))
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    for ring in rings:
        xs = (ring[:, 0] - W0) / RES
        ys = (N0 - ring[:, 1]) / RES
        draw.polygon(list(zip(xs.tolist(), ys.tolist())), fill=1)
    return np.array(img, dtype=bool)


def near_land(land: np.ndarray, mid_lat: float) -> np.ndarray:
    """陸から MARGIN_KM 以内のマス目を True にする(陸を少し太らせる)。"""
    ry = int(np.ceil(MARGIN_KM / (RES * 111)))                       # 南北の幅(マス目の数)
    rx = int(np.ceil(MARGIN_KM / (RES * 111 * np.cos(np.radians(mid_lat)))))  # 東西の幅
    out = land.copy()
    h, w = land.shape
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dy / ry) ** 2 + (dx / rx) ** 2 > 1:                  # 楕円の外は使わない
                continue
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            out[yd, xd] |= land[ys, xs]
    return out


def nearest_open(open_sea, lon, lat):
    """(lon, lat) にいちばん近い開けた海のマス目の中心を返す。なければ None。"""
    c, r = to_px(lon, lat)
    half = int(SEARCH_DEG / RES)
    r0, r1 = max(r - half, 0), min(r + half + 1, open_sea.shape[0])
    c0, c1 = max(c - half, 0), min(c + half + 1, open_sea.shape[1])
    rows, cols = np.nonzero(open_sea[r0:r1, c0:c1])
    if len(rows) == 0:
        return None
    k = np.cos(np.radians(lat))                       # 経度1度の長さは緯度で変わる
    d = np.hypot((cols + c0 - c) * k, rows + r0 - r)
    i = int(np.argmin(d))
    new_lon = W0 + (cols[i] + c0 + 0.5) * RES
    new_lat = N0 - (rows[i] + r0 + 0.5) * RES
    return round(float(new_lon), 4), round(float(new_lat), 4)


def main() -> None:
    pts = pd.read_csv(DATA / "cell_points.csv")
    land = land_raster(load_rings())
    open_sea = ~near_land(land, (LAT_MIN + LAT_MAX) / 2)

    c, r = to_px(pts["plon"].values, pts["plat"].values)
    on_land = land[r, c]
    ok = open_sea[r, c]
    print("印の数", len(pts), "/ 陸の上", int(on_land.sum()),
          "/ 海だが陸に近すぎる(川・入り江・海岸のすぐそば)", int((~on_land & ~ok).sum()))

    moved, failed = [], 0
    for i in np.where(~ok)[0]:
        lon0, lat0 = pts.at[i, "plon"], pts.at[i, "plat"]
        new = nearest_open(open_sea, lon0, lat0)
        if new is None:
            failed += 1
            continue
        pts.at[i, "plon"], pts.at[i, "plat"] = new
        moved.append(float(np.hypot((new[0] - lon0) * 111 * np.cos(np.radians(lat0)),
                                    (new[1] - lat0) * 111)))

    if moved:
        print("動かした印", len(moved), "/ 距離(km): 中央値", round(float(np.median(moved)), 1),
              "/ 最大", round(max(moved), 1))
    c, r = to_px(pts["plon"].values, pts["plat"].values)
    print("動かせなかった印", failed, "/ 確認: まだ陸の上", int(land[r, c].sum()))
    pts.to_csv(DATA / "cell_points.csv", index=False)
    print("保存:", DATA / "cell_points.csv")


if __name__ == "__main__":
    main()
