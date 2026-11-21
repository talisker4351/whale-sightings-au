"""v2: ホエールウォッチングの計画アプリ(Streamlit アプリ本体)

起動:  streamlit run app.py
読むファイル: data/pred_month.csv、data/presence.csv、data/cell_points.csv(pipeline/step6_predict.py が作る)
画面の言葉は、サイドバーの「Language / 言語」で英語と日本語を切り替えられる。
"""
import datetime as dt

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from whale_lib import (CITIES, FORECAST_DAYS, fetch_conditions, in_forecast_window,
                       month_weights, nearest_city, rank_spots, track_record)

st.set_page_config(page_title="Whale Watching in Australia", layout="wide",
                   initial_sidebar_state="collapsed")

# スマホ(幅640px以下)では、タイトルを小さく、上の余白を狭くする
st.markdown("""<style>
@media (max-width: 640px) {
  h1 { font-size: 1.5rem !important; line-height: 1.3 !important; }
  .block-container { padding-top: 1.5rem !important; }
}
</style>""", unsafe_allow_html=True)

MAP_HEIGHT = 380   # 地図の高さ(px)。幅は画面に合わせる
AUSTRALIA = [[-44, 112], [-10, 154]]   # 月ごとの地図で、最初に収める範囲

# ---------------------------------------------------------------
# 画面に出す言葉(英語と日本語)。言葉を直したいときは、ここだけを直す
# ---------------------------------------------------------------
TEXT = {
    "English": {
        "title": "Plan Your Whale Watching in Australia!",
        "intro": "Find when and where humpback whales are easiest to spot along the Australian coast.",
        "how_to": "How to read the numbers",
        "mode": "Mode",
        "mode_month": "Explore by month",
        "mode_trip": "Plan a trip",
        "month": "Month",
        "month_name": lambda m: dt.date(2000, m, 1).strftime("%B"),
        "top10": lambda m: f"Top 10 spots in {dt.date(2000, m, 1):%B}",
        "tip_month": lambda r: f"Index {r['index']:.0f} | records in {r['years']} years",
        "base_city": "Base city",
        "max_km": "Max distance from the city (km, straight line)",
        "dates": "Travel dates",
        "pick_two": "Pick both a start date and an end date.",
        "no_spots": "No spots within that distance. Try a larger distance.",
        "tip_trip": lambda r: f"Score {r['score']:.0f} | records in {r['years']} years | {r['km']:.0f} km from base",
        "km_col": lambda c: f"Distance from {c} (km)",
        "rank_note": lambda what: (f"Spots are ranked by a {what} that predicts how easy it is to spot whales at "
                                   "that time, based on past sightings, season and sea surface temperature. "
                                   "\"Years with records\" is the number of years with at least one actual "
                                   "sighting at that spot and time. Spots that rank high on both are the "
                                   "most reliable picks."),
        "what_month": "spotting index", "what_trip": "score",
        "record_note": lambda n, y0, y1: (f"Years with records: in how many of the {n} years from {y0} to {y1} "
                                          "at least one sighting was recorded at that spot in those months."),
        "best": "Best spots for your trip",
        "forecast": "Sea conditions forecast (top 3 spots)",
        "near": lambda i, near: f"**{i}. near {near}**",
        "no_forecast": "Forecast is unavailable right now.",
        "wave_point": lambda p: f"Wave data from the nearest sea point ({p[0]}, {p[1]}).",
        "rule": "Outlook is a rule of thumb from wave height and wind, not a validated model.",
        "later": f"Forecasts appear when your trip starts within {FORECAST_DAYS} days.",
        "note": ("Index and score show the relative ease of spotting whales estimated from past "
                 "sighting records and sea surface temperature. They are not encounter probabilities. "
                 "Records cluster where observers are. Sources: Atlas of Living Australia (sightings), "
                 "NOAA OISST v2.1 (sea surface temperature), Open-Meteo (forecasts)."),
        "cols": {"rank": "Rank", "years": "Years with records",
                 "glat": "Latitude", "glon": "Longitude",
                 "index": "Spotting index (0-100)", "n_records": "Past records",
                 "near": "Nearest town", "score": "Score", "km": "Distance (km)",
                 "records": "Past records", "date": "Date", "wave_m": "Max wave (m)",
                 "swell_m": "Max swell (m)", "wind_kmh": "Max wind (km/h)", "outlook": "Outlook"},
        "outlook": {"Good": "Good", "Fair": "Fair", "Poor": "Poor", "No data": "No data"},
        "city": lambda c: c,
    },
    "日本語": {
        "title": "オーストラリアでのホエールウォッチングを計画しよう！",
        "intro": "オーストラリアの海岸で、ザトウクジラが見つけやすい時期と場所を調べられます。",
        "how_to": "数字の見方",
        "mode": "表示",
        "mode_month": "月ごとに見る",
        "mode_trip": "旅行を計画する",
        "month": "月",
        "month_name": lambda m: f"{m}月",
        "top10": lambda m: f"{m}月の見やすい場所 トップ10",
        "tip_month": lambda r: f"指数 {r['index']:.0f}|記録があった年 {r['years']}",
        "base_city": "拠点の都市",
        "max_km": "拠点からの最大距離(km、直線)",
        "dates": "旅行の期間",
        "pick_two": "開始日と終了日の両方を選んでください。",
        "no_spots": "その距離の範囲に候補がありません。距離を広げてください。",
        "tip_trip": lambda r: f"スコア {r['score']:.0f}|記録があった年 {r['years']}|拠点から{r['km']:.0f} km",
        "km_col": lambda c: f"{c}からの距離(km)",
        "rank_note": lambda what: (f"順位は、過去の目撃記録・季節・海面水温から、その時期のクジラの見つけやすさを"
                                   f"予測した{what}で決めています。「記録があった年」は、その場所・時期に実際に"
                                   "記録があった年の数です。両方が高い場所ほど、確かな候補です。"),
        "what_month": "見やすさ指数", "what_trip": "スコア",
        "record_note": lambda n, y0, y1: (f"記録があった年: {y0}〜{y1}年の{n}年間のうち、その場所・その時期に"
                                          "1件でも目撃記録があった年の数です。"),
        "best": "旅行におすすめの場所",
        "forecast": "海の状況の予報(上位3か所)",
        "near": lambda i, near: f"**{i}. {near} 付近**",
        "no_forecast": "現在、予報を取得できません。",
        "wave_point": lambda p: f"波のデータは、最も近い海上の地点({p[0]}, {p[1]})のものです。",
        "rule": "見通しは、波の高さと風からの目安です。検証したモデルではありません。",
        "later": f"出発が{FORECAST_DAYS}日以内になると、予報が表示されます。",
        "note": ("指数とスコアは、過去の目撃記録と海面水温から推定した、クジラの見つけやすさの目安です。"
                 "遭遇する確率ではありません。記録は観察する人が多い場所に集まりやすい点に注意してください。"
                 "出典: Atlas of Living Australia(目撃記録)、NOAA OISST v2.1(海面水温)、Open-Meteo(予報)。"),
        "cols": {"rank": "順位", "years": "記録があった年",
                 "glat": "緯度", "glon": "経度",
                 "index": "見やすさ指数(0〜100)", "n_records": "過去の記録数",
                 "near": "近くの町", "score": "スコア", "km": "距離(km)",
                 "records": "過去の記録数", "date": "日付", "wave_m": "最大波高(m)",
                 "swell_m": "最大うねり(m)", "wind_kmh": "最大風速(km/h)", "outlook": "見通し"},
        "outlook": {"Good": "良い", "Fair": "ふつう", "Poor": "悪い", "No data": "データなし"},
        "city": lambda c: {"Sydney": "シドニー", "Brisbane": "ブリスベン", "Hervey Bay": "ハービーベイ",
                           "Cairns": "ケアンズ", "Melbourne": "メルボルン", "Hobart": "ホバート",
                           "Adelaide": "アデレード", "Albany": "アルバニー", "Perth": "パース",
                           "Exmouth": "エクスマウス", "Broome": "ブルーム"}.get(c, c),
    },
}


@st.cache_data
def load_pred() -> pd.DataFrame:
    return pd.read_csv("data/pred_month.csv")


@st.cache_data
def load_presence() -> pd.DataFrame:
    return pd.read_csv("data/presence.csv")


@st.cache_data
def load_points() -> pd.DataFrame:
    try:
        return pd.read_csv("data/cell_points.csv")
    except FileNotFoundError:            # まだ無いときは、マス目の中心に印を置く
        return pd.DataFrame(columns=["glat", "glon", "plat", "plon"])


@st.cache_data(ttl=3600)
def get_conditions(lat, lon, start, end):
    try:
        return fetch_conditions(lat, lon, start, end)
    except Exception:
        return None


def with_points(df, points):
    """地図に印を置く位置(plat, plon)を付ける。記録の平均位置がなければマス目の中心。"""
    df = df.merge(points, on=["glat", "glon"], how="left")
    df["plat"] = df["plat"].fillna(df["glat"])
    df["plon"] = df["plon"].fillna(df["glon"])
    return df


def new_map(location, zoom):
    # 地図の上でマウスのホイールを回しても、ページがスクロールするようにする
    return folium.Map(location=location, zoom_start=zoom, scrollWheelZoom=False)


def add_cells(m, df, value_col, label):
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["plat"], r["plon"]],
            radius=3 + r[value_col] / 10,
            color="#0d6b78", fill=True, fill_opacity=0.5,
            tooltip=label(r),
        ).add_to(m)
    return m


def add_numbers(m, df):
    """上位の場所に、表の順位と同じ番号を地図上に書く。"""
    for _, r in df.iterrows():
        html = ('<div style="width:22px;height:22px;border-radius:11px;background:#0b3d45;'
                'color:#fff;font:bold 12px sans-serif;display:flex;align-items:center;'
                f'justify-content:center;border:2px solid #fff">{int(r["rank"])}</div>')
        folium.Marker([r["plat"], r["plon"]],
                      icon=folium.DivIcon(html=html, icon_size=(22, 22), icon_anchor=(11, 11))
                      ).add_to(m)
    return m


def with_record(df, months, presence):
    """過去の実績(記録があった年の数)を「15 / 26」の形で付ける。"""
    seen, n, y0, y1 = track_record(presence, months)
    df = df.merge(seen, on=["glat", "glon"], how="left")
    df["years"] = df["years_seen"].fillna(0).astype(int).astype(str) + f" / {n}"
    return df, n, y0, y1


pred = load_pred()
presence = load_presence()
points = load_points()
lang = st.radio("Language / 言語", ["English", "日本語"], horizontal=True)
T = TEXT[lang]

st.title(T["title"])
st.caption(T["intro"])
mode = st.radio(T["mode"], ["month", "trip"], horizontal=True,
                        format_func=lambda k: T["mode_month"] if k == "month" else T["mode_trip"],
                        key="mode")

if mode == "month":
    month = st.selectbox(T["month"], list(range(1, 13)), index=6,
                                 format_func=T["month_name"], key="month")
    sub, n, y0, y1 = with_record(pred[pred["month"] == month], [month], presence)
    sub = with_points(sub, points)
    top = sub.sort_values("index", ascending=False).head(10).reset_index(drop=True)
    top["rank"] = top.index + 1
    top["near"] = [nearest_city(a, b) for a, b in zip(top["plat"], top["plon"])]
    m = new_map([-28, 134], 4)
    m.fit_bounds(AUSTRALIA)
    add_cells(m, sub[sub["index"] >= 5], "index", T["tip_month"])  # 指数5未満は省く
    add_numbers(m, top)
    st_folium(m, use_container_width=True, height=MAP_HEIGHT, returned_objects=[])
    st.subheader(T["top10"](month))
    st.dataframe(top[["rank", "near", "index", "years"]].rename(columns=T["cols"]),
                 hide_index=True)
    with st.expander(T["how_to"]):
        st.write(T["rank_note"](T["what_month"]))
        st.write(T["record_note"](n, y0, y1))
        st.write(T["note"])

else:
    city = st.selectbox(T["base_city"], list(CITIES), format_func=T["city"],
                                key="city")
    max_km = st.slider(T["max_km"], 50, 1500, 300, step=50, key="max_km")
    today = dt.date.today()
    dates = st.date_input(
        T["dates"], (today + dt.timedelta(days=3), today + dt.timedelta(days=9)), key="dates")
    if not isinstance(dates, (tuple, list)) or len(dates) != 2:
        st.info(T["pick_two"])
        st.stop()
    start, end = dates

    lat0, lon0 = CITIES[city]
    weights = month_weights(start, end)
    top = rank_spots(pred, lat0, lon0, max_km, weights, top=10)
    if top.empty:
        st.warning(T["no_spots"])
        st.stop()

    m = new_map([lat0, lon0], 6)
    folium.Marker([lat0, lon0], tooltip=T["city"](city)).add_to(m)
    top, n, y0, y1 = with_record(top, list(weights), presence)
    top = with_points(top, points)
    top["rank"] = top.index + 1
    add_cells(m, top, "score", T["tip_trip"])
    add_numbers(m, top)
    # 拠点の都市と上位10か所が、すべて画面に収まるように縮尺を合わせる
    lats = list(top["plat"]) + [lat0]
    lons = list(top["plon"]) + [lon0]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    st_folium(m, use_container_width=True, height=MAP_HEIGHT, returned_objects=[])

    st.subheader(T["best"])
    cols = {**T["cols"], "km": T["km_col"](T["city"](city))}   # 距離の列名に拠点の都市名を入れる
    st.dataframe(top[["rank", "near", "score", "years", "km"]]
                    .rename(columns=cols), hide_index=True)
    with st.expander(T["how_to"]):
        st.write(T["rank_note"](T["what_trip"]))
        st.write(T["record_note"](n, y0, y1))
        st.write(T["note"])

    st.subheader(T["forecast"])
    if in_forecast_window(start, today):
        fend = min(end, today + dt.timedelta(days=FORECAST_DAYS - 1))
        for i, r in top.head(3).iterrows():
            st.markdown(T["near"](i + 1, r["near"]))
            cond = get_conditions(r["glat"], r["glon"], start, fend)
            if cond is None:
                st.write(T["no_forecast"])
            else:
                shown = cond.copy()
                shown["outlook"] = shown["outlook"].map(T["outlook"])
                st.dataframe(shown.rename(columns=T["cols"]), hide_index=True)
                wp = cond.attrs.get("wave_point")
                if wp and (wp[0], wp[1]) != (r["glat"], r["glon"]):
                    st.caption(T["wave_point"](wp))
        st.caption(T["rule"])
    else:
        st.info(T["later"])

