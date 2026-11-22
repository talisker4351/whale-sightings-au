"""アプリ(v2)が使う計算と、海況予報の取得。画面のコードと分けて、単独で確かめられるようにしている。"""
import datetime as dt
import math

import numpy as np
import pandas as pd
import requests

# 拠点にする都市(緯度, 経度)
CITIES = {
    "Sydney": (-33.87, 151.21),
    "Brisbane": (-27.47, 153.03),
    "Gold Coast": (-28.02, 153.40),
    "Hervey Bay": (-25.29, 152.85),
    "Cairns": (-16.92, 145.77),
    "Melbourne": (-37.81, 144.96),
    "Hobart": (-42.88, 147.33),
    "Adelaide": (-34.93, 138.60),
    "Albany": (-35.02, 117.88),
    "Perth": (-31.95, 115.86),
    "Exmouth": (-21.93, 114.13),
    "Broome": (-17.96, 122.24),
}

# 「近くの町」の表示に使う海沿いの町と観察地(緯度, 経度はおおよその値)
TOWNS = {
    **CITIES,
    "Coffs Harbour": (-30.30, 153.11),
    "Byron Bay": (-28.64, 153.61),
    "Point Lookout": (-27.43, 153.54),
    "Sunshine Coast": (-26.68, 153.12),
    "Port Macquarie": (-31.43, 152.91),
    "Port Stephens": (-32.72, 152.14),
    "Jervis Bay": (-35.04, 150.67),
    "Eden": (-37.07, 149.90),
    "Phillip Island": (-38.45, 145.24),
    "Warrnambool": (-38.38, 142.48),
    "Victor Harbor": (-35.55, 138.62),
    "Head of Bight": (-31.48, 131.10),
    "Esperance": (-33.86, 121.89),
    "Augusta": (-34.31, 115.16),
    "Busselton": (-33.65, 115.35),
    "Geraldton": (-28.78, 114.61),
    "Kalbarri": (-27.71, 114.16),
    "Coral Bay": (-23.14, 113.77),
    "Townsville": (-19.26, 146.82),
    "Airlie Beach": (-20.27, 148.72),
    "Wollongong": (-34.43, 150.89),
    "Kiama": (-34.67, 150.85),
    "Ulladulla": (-35.36, 150.47),
    "Batemans Bay": (-35.71, 150.18),
    "Narooma": (-36.22, 150.13),
    "Merimbula": (-36.89, 149.91),
    "Newcastle": (-32.93, 151.78),
    "Forster": (-32.18, 152.51),
    "Yamba": (-29.44, 153.36),
    "Noosa": (-26.39, 153.09),
    "Bundaberg": (-24.87, 152.35),
    "Gladstone": (-23.84, 151.26),
    "Mackay": (-21.14, 149.19),
    "Port Douglas": (-16.48, 145.47),
    "Lorne": (-38.54, 143.98),
    "Portland": (-38.34, 141.60),
    "Kangaroo Island": (-35.66, 137.64),
    "Port Lincoln": (-34.72, 135.86),
    "Ceduna": (-32.13, 133.68),
    "Bremer Bay": (-34.39, 119.38),
    "Dunsborough": (-33.61, 115.10),
    "Fremantle": (-32.06, 115.74),
    "Jurien Bay": (-30.31, 115.04),
    "Denham (Shark Bay)": (-25.93, 113.53),
    "Carnarvon": (-24.88, 113.66),
    "Dampier": (-20.66, 116.71),
    "Port Hedland": (-20.31, 118.58),
    "Darwin": (-12.46, 130.84),
}

# 波のデータが空だったときに試す、少し沖の地点(緯度・経度のずらし幅。0.25度は約25km)
OFFSHORE_STEPS = [(0, 0), (0, 0.25), (0, -0.25), (0.25, 0), (-0.25, 0), (0, 0.5), (0, -0.5)]

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_DAYS = 7


def haversine_km(lat1, lon1, lat2, lon2):
    """2点間の直線距離(km)。地球を球とみなす。道路の距離ではない。"""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def month_weights(start: dt.date, end: dt.date) -> dict:
    """旅行期間のうち、各月が占める日数の割合。例: 7/25〜8/3 → {7: 0.7, 8: 0.3}"""
    if end < start:
        raise ValueError("end must be on or after start")
    days = pd.date_range(start, end, freq="D")
    return days.month.value_counts(normalize=True).sort_index().to_dict()


def nearest_city(lat: float, lon: float) -> str:
    """いちばん近い町の名前。"""
    names = list(TOWNS)
    d = [float(haversine_km(lat, lon, *TOWNS[n])) for n in names]
    return names[int(np.argmin(d))]


def rank_spots(pred: pd.DataFrame, base_lat: float, base_lon: float,
               max_km: float, weights: dict, top: int = 10) -> pd.DataFrame:
    """旅行期間で重みを付けた指数の合計が高いマスを、距離の上限内で順位付けする。

    pred は pred_month.csv(glat, glon, month, index, n_records)。
    予測がない月は0として扱う。
    """
    sub = pred[pred["month"].isin(list(weights))].copy()
    sub["w"] = sub["month"].map(weights)
    sub["wi"] = sub["index"] * sub["w"]
    agg = (sub.groupby(["glat", "glon"])
              .agg(score=("wi", "sum"), records=("n_records", "sum"))
              .reset_index())
    agg["km"] = haversine_km(base_lat, base_lon, agg["glat"], agg["glon"]).round(0)
    agg = agg[agg["km"] <= max_km]
    agg = agg.sort_values("score", ascending=False).head(top).reset_index(drop=True)
    agg["score"] = agg["score"].round(1)
    agg["near"] = [nearest_city(a, b) for a, b in zip(agg["glat"], agg["glon"])]
    return agg


def track_record(presence: pd.DataFrame, months) -> tuple[pd.DataFrame, int, int, int]:
    """指定した月のどれかに記録があった年の数を、マスごとに数える。

    presence は presence.csv(glat, glon, year, month)。
    戻り値: (glat, glon, years_seen の表, 対象の年数, 最初の年, 最後の年)
    """
    y0, y1 = int(presence["year"].min()), int(presence["year"].max())
    seen = (presence[presence["month"].isin(list(months))]
              .groupby(["glat", "glon"])["year"].nunique()
              .rename("years_seen").reset_index())
    return seen, y1 - y0 + 1, y0, y1


def in_forecast_window(start: dt.date, today: dt.date, days: int = FORECAST_DAYS) -> bool:
    return 0 <= (start - today).days < days


def viewing_label(wave_m, wind_kmh) -> str:
    """波と風からの目安。検証したモデルではなく、経験的な区切り。"""
    if wave_m is None or wind_kmh is None or (isinstance(wave_m, float) and math.isnan(wave_m)) \
            or (isinstance(wind_kmh, float) and math.isnan(wind_kmh)):
        return "No data"
    if wave_m <= 1.5 and wind_kmh <= 20:
        return "Good"
    if wave_m <= 2.5 and wind_kmh <= 30:
        return "Fair"
    return "Poor"


def fetch_conditions(lat: float, lon: float, start: dt.date, end: dt.date,
                     timeout: int = 10) -> pd.DataFrame:
    """Open-Meteoから日ごとの最大波高・うねり・最大風速を取る。APIキーは不要。"""
    dates = {"timezone": "auto", "start_date": start.isoformat(), "end_date": end.isoformat()}
    # 海岸のすぐそばは波のデータが空のことがあるので、少しずつ沖へずらして取り直す
    md, used = None, (lat, lon)
    for dlat, dlon in OFFSHORE_STEPS:
        p = (round(lat + dlat, 4), round(lon + dlon, 4))
        m = requests.get(MARINE_URL, timeout=timeout,
                         params={**dates, "latitude": p[0], "longitude": p[1],
                                 "cell_selection": "sea",
                                 "daily": "wave_height_max,swell_wave_height_max"})
        m.raise_for_status()
        d = m.json()["daily"]
        if any(v is not None for v in d["wave_height_max"]):
            md, used = d, p
            break
        if md is None:
            md = d          # すべて空なら、最初の結果(空)をそのまま使う
    w = requests.get(WEATHER_URL, timeout=timeout,
                     params={**dates, "latitude": lat, "longitude": lon,
                             "daily": "wind_speed_10m_max"})
    w.raise_for_status()
    wd = w.json()["daily"]
    df = pd.DataFrame({"date": md["time"],
                       "wave_m": md["wave_height_max"],
                       "swell_m": md["swell_wave_height_max"]})
    df = df.merge(pd.DataFrame({"date": wd["time"], "wind_kmh": wd["wind_speed_10m_max"]}),
                  on="date", how="left")
    df = df.astype({"wave_m": float, "swell_m": float, "wind_kmh": float})
    df["outlook"] = [viewing_label(a, b) for a, b in zip(df["wave_m"], df["wind_kmh"])]
    df.attrs["wave_point"] = used      # 波を取った地点(沖へずらした場合は元の地点と違う)
    return df
