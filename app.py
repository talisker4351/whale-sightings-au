"""v2: ホエールウォッチングの計画アプリ(Streamlit アプリ本体)

起動:  streamlit run app.py
読むファイル: data/pred_month.csv、data/presence.csv、data/cell_points.csv(pipeline/step6_predict.py が作る)
画面の言葉は、画面上部の「Language / 言語」で英語と日本語を切り替えられる。
"""
import datetime as dt
import json
import os

import folium
import pandas as pd
import streamlit as st
from jinja2 import Template
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
# 泳ぐクジラ(飾り)。絵は自作のSVG、動きはCSSのアニメーションだけ(JavaScriptなし)。
# 「動きを減らす」設定の人には、動かさずに表示する。
# ---------------------------------------------------------------
# クジラの絵。左右反転(scale(-1,1))して、頭を右(進む方向)に向けている
WHALE_BODY = (
    '<g transform="translate(80,0) scale(-1,1)">'
    '<g class="wh-spout"><path d="M22 14 C20 6 16 3 13 2 M22 14 C22 5 22 2 22 0 M22 14 C24 6 28 3 31 2" '
    'stroke="#85B7EB" stroke-width="2" fill="none" stroke-linecap="round"/></g>'
    '<path d="M6 26 C6 16 22 12 38 14 C52 16 60 22 66 24 L76 16 C77 22 75 27 70 29 L76 36 '
    'C70 35 66 33 63 31 C50 37 22 38 10 33 C7 31 6 29 6 26Z" fill="#185FA5"/>'
    '<path d="M10 30 C22 34 44 34 58 30" stroke="#B5D4F4" stroke-width="1.5" fill="none"/>'
    '<circle cx="16" cy="24" r="1.6" fill="#E6F1FB"/>'
    '<path d="M30 31 C33 36 38 38 42 37" stroke="#0C447C" stroke-width="2" fill="none" '
    'stroke-linecap="round"/></g>'
)

# タップしたときに吹く、大きめの潮(自動の潮吹きとは別の部品)
TAP_SPOUT = ('<g class="wh-tapspout"><path d="M22 14 C19 5 14 2 10 1 M22 14 C22 4 22 1 22 -2 '
             'M22 14 C25 5 30 2 34 1" stroke="#85B7EB" stroke-width="2.5" fill="none" '
             'stroke-linecap="round"/></g>')
FLIP = '<g transform="translate(80,0) scale(-1,1)">'
TAP_KEYS = ("{0%{opacity:0;transform:scale(.4,.2)}25%{opacity:1;transform:scale(1.3,1.5)}"
            "100%{opacity:0;transform:scale(1.4,1.8)}}")

# A: タイトルの下の細い海の帯を、クジラが左から右へ泳ぐ(約14秒で横切り、途中で潮を吹く)
WHALE_BANNER = (
    "<style>"
    ".wh-sea{position:relative;height:56px;overflow:hidden;border-radius:12px;"
    "background:rgba(55,138,221,0.15);margin:0 0 0.5rem}"
    ".wh-wave{position:absolute;left:0;bottom:0;width:200%;height:14px;opacity:.5;"
    "animation:wh-flow 8s linear infinite}"
    ".wh-swim{position:absolute;bottom:8px;left:-90px;width:80px;animation:wh-swim 14s linear infinite}"
    ".wh-swim svg{display:block;animation:wh-bob 2.4s ease-in-out infinite}"
    ".wh-sea .wh-spout{transform-box:view-box;transform-origin:22px 14px;opacity:0;"
    "animation:wh-spout 14s ease-out infinite}"
    "@keyframes wh-flow{to{transform:translateX(-50%)}}"
    "@keyframes wh-swim{to{left:110%}}"
    "@keyframes wh-bob{50%{transform:translateY(-4px) rotate(2deg)}}"
    "@keyframes wh-spout{0%,38%{opacity:0;transform:scaleY(.2)}42%{opacity:1;transform:scaleY(1)}"
    "50%,100%{opacity:0;transform:scaleY(1.1)}}"
    # タップ: 見えないチェックボックスを、帯(label)のタップで入・切する。
    # 入と切で別の名前の動き(中身は同じ)を当てるので、タップのたびに潮吹きが最初から再生される
    ".wh-sea{display:block;cursor:pointer;-webkit-tap-highlight-color:transparent}"
    ".wh-tap{position:absolute;opacity:0;pointer-events:none}"
    ".wh-sea .wh-tapspout{transform-box:view-box;transform-origin:22px 14px;opacity:0}"
    ".wh-tap:checked~.wh-swim .wh-tapspout{animation:wh-tapa .9s ease-out}"
    ".wh-tap:not(:checked)~.wh-swim .wh-tapspout{animation:wh-tapb .9s ease-out}"
    "@keyframes wh-tapa" + TAP_KEYS + "@keyframes wh-tapb" + TAP_KEYS +
    "@media (prefers-reduced-motion: reduce){.wh-wave,.wh-swim,.wh-swim svg,.wh-sea .wh-spout"
    "{animation:none}.wh-swim{left:40%}}"
    "</style>"
    '<label class="wh-sea" title="Tap the whale!">'
    '<input type="checkbox" class="wh-tap" aria-label="Make the whale spout">'
    '<svg class="wh-wave" viewBox="0 0 800 14" preserveAspectRatio="none" aria-hidden="true">'
    '<path d="M0 7 Q25 0 50 7' + "".join(f" T{x} 7" for x in range(100, 801, 50)) +
    ' V14 H0Z" fill="#378ADD"/></svg>'
    '<span class="wh-swim" style="display:block"><svg viewBox="0 0 80 40" width="80" height="40" '
    'aria-hidden="true">' + WHALE_BODY.replace(FLIP, FLIP + TAP_SPOUT) +
    "</svg></span></label>"
)


def map_whale(rank):
    """B: 地図の1〜3位の番号の横に置く小さなクジラ。数秒おきに潮を吹く(順位ごとに時間をずらす)。"""
    delay = (rank - 1) * 1.3
    return (
        "<style>"
        ".wh-mini{position:absolute;left:16px;top:-14px;width:30px;height:15px;cursor:pointer;"
        "overflow:visible}"
        ".wh-mini .wh-tapspout{transform-box:view-box;transform-origin:22px 14px;opacity:0}"
        ".wh-mini .wh-tapspout.go{animation:wh-mtap .9s ease-out}"   # .go は TapSpout が付け外しする
        "@keyframes wh-mtap" + TAP_KEYS +
        ".wh-mini .wh-spout{transform-box:view-box;transform-origin:22px 14px;opacity:0;"
        "animation:wh-mspout 4s ease-out infinite}"
        "@keyframes wh-mspout{0%,60%{opacity:0;transform:scaleY(.2)}70%{opacity:1;transform:scaleY(1)}"
        "85%,100%{opacity:0;transform:scaleY(1.1)}}"
        "@media (prefers-reduced-motion: reduce){.wh-mini .wh-spout{animation:none;opacity:1}}"
        "</style>"
        f'<svg class="wh-mini" viewBox="0 0 80 40">'
        + WHALE_BODY.replace(FLIP, FLIP + TAP_SPOUT)
                    .replace('class="wh-spout"', f'class="wh-spout" style="animation-delay:{delay}s"')
                    .replace('stroke="#85B7EB" stroke-width="2"', 'stroke="#ffffff" stroke-width="4"')
                    .replace('stroke="#85B7EB" stroke-width="2.5"', 'stroke="#ffffff" stroke-width="5"')
        # ↑ 地図の水色の海でも見えるよう、潮を白く太くする
        + "</svg>"
    )

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
        "tip_month": lambda r: f"Index {r['index']:.0f} | seen in {r['years']} years",
        "base_city": "Base city",
        "max_km": "How far from your base? (km)",
        "max_km_note": "Straight-line distance. Driving distance will be longer.",
        "dates": "Travel dates",
        "pick_two": "Pick both a start date and an end date.",
        "no_spots": "No spots within that distance. Try a larger distance.",
        "tip_trip": lambda r: (f"Score {r['score']:.0f} | seen in {r['years']} years | "
                               f"{r['km']:.0f} km from base"),
        "km_col": lambda c: f"km from {c}",
        # 見出しと表の間に出す、短い説明(半角スペース2つ+\n で改行)
        "explain_month": lambda mon, n: (
            "Index: closer to 100 = easier to spot (predicted from past sightings, season "
            "and sea temperature; 100 is the best)  \n"
            f"Years seen this season: in how many of the {n} years whales were recorded "
            f"near this spot in {mon}"),
        "explain_trip": lambda mon, n: (
            "Score: closer to 100 = easier to spot (the index adjusted to your trip dates)  \n"
            f"Years seen this season: in how many of the {n} years whales were recorded "
            f"near this spot in {mon}"),
        "months_join": lambda ms: " or ".join(dt.date(2000, m, 1).strftime("%B") for m in ms),
        "years_fmt": "%d",
        "rank_note": lambda what: (f"Spots are ranked by the {what}, predicted from past sightings, season "
                                   "and sea surface temperature. Spots that are also high on \"Years seen "
                                   "this season\" are the most reliable picks."),
        "what_month": "spotting index", "what_trip": "score",
        "record_note": lambda n, y0, y1: (f"Years seen this season: in how many of the {n} years from {y0} "
                                          f"to {y1} at least one sighting was recorded in that area (about "
                                          "25 km square) in those months. A year without records does not "
                                          "mean there were no whales; nobody may have reported them."),
        "trip_note": ("The score combines the index of each month in your trip, weighted by the "
                      "number of days (e.g. 25 Nov to 4 Dec = 6 days of November and 4 of December)."),
        "best": "Best spots for your trip",
        "forecast": "Sea conditions forecast (top 3 spots)",
        "near": lambda i, near: f"**{i}. near {near}**",
        "no_forecast": "Forecast is unavailable right now.",
        "wave_point": lambda p: f"Wave data from the nearest sea point ({p[0]}, {p[1]}).",
        "rule": ("Wave and wind are daily maximums. Outlook is a rule of thumb from wave height "
                 "and wind, not a validated model."),
        "later": f"Forecasts appear when your trip starts within {FORECAST_DAYS} days.",
        "note": ("Index (0-100; the best spot and month of the year = 100) and score show the "
                 "relative ease of spotting whales estimated from past sighting records and sea "
                 "surface temperature. They are not encounter probabilities. "
                 "Records cluster where observers are. Sources: Atlas of Living Australia (sightings), "
                 "NOAA OISST v2.1 (sea surface temperature), Open-Meteo (forecasts)."),
        # 列名はスマホの幅に収まるよう短くする(意味は「数字の見方」で説明する)
        "cols": {"rank": "Rank", "years": "Years seen this season",
                 "glat": "Latitude", "glon": "Longitude",
                 "index": "Index", "n_records": "Records",
                 "near": "Town", "score": "Score", "km": "km",
                 "records": "Records", "date": "Date", "wave_m": "Wave (m)",
                 "swell_m": "Swell (m)", "wind_kmh": "Wind (km/h)", "outlook": "Outlook"},
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
        "tip_month": lambda r: f"指数 {r['index']:.0f}|この時期に見られた年 {r['years']}年",
        "base_city": "拠点の都市",
        "max_km": "拠点から何kmまで探しますか？",
        "max_km_note": "地図上のまっすぐな距離です。車での道のりは、これより長くなります。",
        "dates": "旅行の期間",
        "pick_two": "開始日と終了日の両方を選んでください。",
        "no_spots": "その距離の範囲に候補がありません。距離を広げてください。",
        "tip_trip": lambda r: (f"スコア {r['score']:.0f}|この時期に見られた年 {r['years']}年|"
                               f"拠点から{r['km']:.0f} km"),
        "km_col": lambda c: f"{c}から(km)",
        "explain_month": lambda mon, n: (
            "指数：100に近いほど見つけやすい(過去の記録・季節・海水温から予測。最高は100)  \n"
            f"この時期に見られた年：{n}年間で、{mon}にこの近くで目撃があった年の数"),
        "explain_trip": lambda mon, n: (
            "スコア：100に近いほど見つけやすい(指数を旅行の期間に合わせたもの)  \n"
            f"この時期に見られた年：{n}年間で、{mon}にこの近くで目撃があった年の数"),
        "months_join": lambda ms: "か".join(f"{m}月" for m in ms),   # 例: 11月か12月
        "years_fmt": "%d年",                                          # 表の「11年」
        "rank_note": lambda what: (f"順位は、過去の目撃記録・季節・海面水温から予測した{what}の高い順です。"
                                   "「この時期に見られた年」も多い場所ほど、確かな候補です。"),
        "what_month": "見やすさ指数", "what_trip": "スコア",
        "record_note": lambda n, y0, y1: (f"この時期に見られた年: {y0}〜{y1}年の{n}年間のうち、その場所"
                                          "(約25km四方)・その月に1件でも目撃記録があった年の数です。"
                                          "記録がない年は、クジラがいなかったとは限りません"
                                          "(誰も記録しなかった年も含みます)。"),
        "trip_note": ("スコアは、旅行の期間に含まれる各月の指数を、日数の割合で合わせたものです"
                      "(例: 11/25〜12/4なら、11月を6日分・12月を4日分)。"),
        "best": "旅行におすすめの場所",
        "forecast": "海の状況の予報(上位3か所)",
        "near": lambda i, near: f"**{i}. {near} 付近**",
        "no_forecast": "現在、予報を取得できません。",
        "wave_point": lambda p: f"波のデータは、最も近い海上の地点({p[0]}, {p[1]})のものです。",
        "rule": "波と風は、その日の最大値です。見通しは、波の高さと風からの目安です。検証したモデルではありません。",
        "later": f"出発が{FORECAST_DAYS}日以内になると、予報が表示されます。",
        "note": ("指数(0〜100。1年でいちばん見つけやすい場所・月が100)とスコアは、"
                 "過去の目撃記録と海面水温から推定した、クジラの見つけやすさの目安です。"
                 "遭遇する確率ではありません。記録は観察する人が多い場所に集まりやすい点に注意してください。"
                 "出典: Atlas of Living Australia(目撃記録)、NOAA OISST v2.1(海面水温)、Open-Meteo(予報)。"),
        "cols": {"rank": "順位", "years": "この時期に見られた年",
                 "glat": "緯度", "glon": "経度",
                 "index": "指数", "n_records": "記録数",
                 "near": "近くの町", "score": "スコア", "km": "km",
                 "records": "記録数", "date": "日付", "wave_m": "波(m)",
                 "swell_m": "うねり(m)", "wind_kmh": "風(km/h)", "outlook": "見通し"},
        "outlook": {"Good": "良い", "Fair": "ふつう", "Poor": "悪い", "No data": "データなし"},
        "city": lambda c: {"Sydney": "シドニー", "Brisbane": "ブリスベン", "Gold Coast": "ゴールドコースト",
                           "Hervey Bay": "ハービーベイ",
                           "Cairns": "ケアンズ", "Melbourne": "メルボルン", "Hobart": "ホバート",
                           "Adelaide": "アデレード", "Albany": "アルバニー", "Perth": "パース",
                           "Exmouth": "エクスマウス", "Broome": "ブルーム"}.get(c, c),
    },
}


def file_stamp(path: str) -> float:
    """ファイルの更新時刻。データを差し替えたら、覚えておいた古い表を使わず読み直すために使う。"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


# 読み込んだ表は覚えておく(st.cache_data)。ただし覚える目印に「ファイルの更新時刻」を入れて、
# GitHub でCSVだけを差し替えたときも、新しいファイルを読み直すようにする
@st.cache_data
def load_csv(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_pred() -> pd.DataFrame:
    return load_csv("data/pred_month.csv", file_stamp("data/pred_month.csv"))


def load_presence() -> pd.DataFrame:
    return load_csv("data/presence.csv", file_stamp("data/presence.csv"))


def load_points() -> pd.DataFrame:
    path = "data/cell_points.csv"
    if not os.path.exists(path):         # まだ無いときは、マス目の中心に印を置く
        return pd.DataFrame(columns=["glat", "glon", "plat", "plon"])
    return load_csv(path, file_stamp(path))


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
    # scrollWheelZoom=False: 地図の上でマウスのホイールを回しても、ページがスクロールする
    # zoomSnap=0.25: 縮尺を細かく選べるようにして、狭いスマホの画面でも範囲がぴったり収まるようにする
    return folium.Map(location=location, zoom_start=zoom, scrollWheelZoom=False,
                      zoomSnap=0.25, zoomDelta=0.5)


class Refit(folium.MacroElement):
    """地図の幅が決まった後に、もう一度「範囲に合わせて縮尺を決め直す」JavaScript を足す。

    スマホでは、地図を作った瞬間の幅と、実際に表示される幅がずれるため、
    fit_bounds だけでは端の印が画面からはみ出す。少し待ってから合わせ直す。
    """
    _template = Template("""
{% macro script(this, kwargs) %}
(function () {
  var target = {{ this._parent.get_name() }};
  var area = {{ this.area }};
  function refit() {
    target.invalidateSize();
    target.fitBounds(area, {padding: [{{ this.pad }}, {{ this.pad }}]});
  }
  setTimeout(refit, 150);
  setTimeout(refit, 700);
  window.addEventListener("resize", refit);
})();
{% endmacro %}
""")

    def __init__(self, bounds, pad=10):
        super().__init__()
        self._name = "Refit"
        self.area = json.dumps([[float(a), float(b)] for a, b in bounds])
        self.pad = int(pad)


class TapSpout(folium.MacroElement):
    """地図の1〜3位の番号かクジラをタップすると、そのクジラが潮を吹く JavaScript を足す。"""
    _template = Template("""
{% macro script(this, kwargs) %}
(function () {
  var target = {{ this._parent.get_name() }};
  target.getContainer().addEventListener("click", function (e) {
    var badge = e.target.closest ? e.target.closest(".wh-badge") : null;
    if (!badge) return;
    var spout = badge.querySelector(".wh-tapspout");
    if (!spout) return;
    spout.classList.remove("go");
    void spout.getBoundingClientRect();
    spout.classList.add("go");
  }, true);
})();
{% endmacro %}
""")

    def __init__(self):
        super().__init__()
        self._name = "TapSpout"


def fit(m, bounds, pad=10):
    """地図を bounds([[南, 西], [北, 東]])の範囲に合わせる。表示後にも合わせ直す。"""
    m.fit_bounds(bounds, padding=(pad, pad))
    Refit(bounds, pad).add_to(m)


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
        rank = int(r["rank"])
        html = ('<div class="wh-badge" style="position:relative;width:22px;height:22px;border-radius:11px;'
                'background:#0b3d45;color:#fff;font:bold 12px sans-serif;display:flex;'
                'align-items:center;justify-content:center;border:2px solid #fff">'
                f'{rank}{map_whale(rank) if rank <= 3 else ""}</div>')   # 1〜3位にはクジラを添える
        folium.Marker([r["plat"], r["plon"]],
                      icon=folium.DivIcon(html=html, icon_size=(22, 22), icon_anchor=(11, 11))
                      ).add_to(m)
    TapSpout().add_to(m)          # タップで潮吹き
    return m


def with_record(df, months, presence):
    """この時期に見られた年の数を、数字のまま付ける(表で横棒グラフにするため)。"""
    seen, n, y0, y1 = track_record(presence, months)
    df = df.merge(seen, on=["glat", "glon"], how="left")
    df["years"] = df["years_seen"].fillna(0).astype(int)
    return df, n, y0, y1


def show_table(df, cols, n):
    """表を出す。「この時期に見られた年」は、最大n年の横棒グラフにする。"""
    name = cols["years"]
    st.dataframe(df.rename(columns=cols), hide_index=True,
                 column_config={name: st.column_config.ProgressColumn(
                     name, format=T["years_fmt"], min_value=0, max_value=n)})


pred = load_pred()
presence = load_presence()
points = load_points()
lang = st.radio("Language / 言語", ["English", "日本語"], horizontal=True)
T = TEXT[lang]

st.title(T["title"])
st.markdown(WHALE_BANNER, unsafe_allow_html=True)   # A: タイトルの下を泳ぐクジラ
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
    fit(m, AUSTRALIA, pad=5)
    add_cells(m, sub[sub["index"] >= 5], "index", T["tip_month"])  # 指数5未満は省く
    add_numbers(m, top)
    st_folium(m, use_container_width=True, height=MAP_HEIGHT, returned_objects=[])
    st.subheader(T["top10"](month))
    st.caption(T["explain_month"](T["month_name"](month), n))
    show_table(top[["rank", "near", "index", "years"]], T["cols"], n)
    with st.expander(T["how_to"]):
        st.write(T["rank_note"](T["what_month"]))
        st.write(T["record_note"](n, y0, y1))
        st.write(T["note"])

else:
    city = st.selectbox(T["base_city"], list(CITIES), format_func=T["city"],
                                key="city")
    max_km = st.slider(T["max_km"], 50, 500, 300, step=50, key="max_km")   # 最大500km(普通の旅行で動ける範囲)
    st.caption(T["max_km_note"])
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
    # 拠点の都市は赤い丸(番号の印と重ならないよう、ピンではなく小さな丸にする)
    folium.CircleMarker([lat0, lon0], radius=7, color="#c0392b", weight=3, fill=True,
                        fill_color="#ffffff", fill_opacity=1, tooltip=T["city"](city)).add_to(m)
    top, n, y0, y1 = with_record(top, list(weights), presence)
    top = with_points(top, points)
    top["rank"] = top.index + 1
    add_cells(m, top, "score", T["tip_trip"])
    add_numbers(m, top)
    # 拠点の都市と上位10か所が、すべて画面に収まるように縮尺を合わせる
    lats = list(top["plat"]) + [lat0]
    lons = list(top["plon"]) + [lon0]
    fit(m, [[min(lats), min(lons)], [max(lats), max(lons)]], pad=20)
    st_folium(m, use_container_width=True, height=MAP_HEIGHT, returned_objects=[])

    st.subheader(T["best"])
    trip_months = list(dict.fromkeys(pd.date_range(start, end).month))   # 旅行の順(12月→1月)に並べる
    st.caption(T["explain_trip"](T["months_join"](trip_months), n))
    cols = {**T["cols"], "km": T["km_col"](T["city"](city))}   # 距離の列名に拠点の都市名を入れる
    show_table(top[["rank", "near", "score", "years", "km"]], cols, n)
    with st.expander(T["how_to"]):
        st.write(T["rank_note"](T["what_trip"]))
        st.write(T["record_note"](n, y0, y1))
        st.write(T["trip_note"])
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
                # スマホの幅に収めるため、日付は「11/24」の形にし、うねり(波の一部)の列は省く
                shown["date"] = pd.to_datetime(shown["date"]).dt.strftime("%m/%d")
                shown = shown.drop(columns=["swell_m"], errors="ignore")
                st.dataframe(shown.rename(columns=T["cols"]), hide_index=True)
                wp = cond.attrs.get("wave_point")
                if wp and (wp[0], wp[1]) != (r["glat"], r["glon"]):
                    st.caption(T["wave_point"](wp))
        st.caption(T["rule"])
    else:
        st.info(T["later"])

