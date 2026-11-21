"""v0: 月を選ぶと、過去の目撃件数が地図に出るアプリ(機械学習なし)。参考用に残している。

起動:  streamlit run app_v0.py
"""
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


@st.cache_data
def load():
    return pd.read_csv("data/grid_month.csv")


def make_map(grid, month):
    m = folium.Map(location=[-28, 134], zoom_start=4)
    sub = grid[grid["month"] == month]
    for _, r in sub.iterrows():
        folium.CircleMarker(
            location=[r["glat"], r["glon"]],
            radius=2 + r["n"] ** 0.5,
            fill=True, fill_opacity=0.5,
            tooltip=str(int(r["n"])) + " sighting records",
        ).add_to(m)
    return m


st.title("Whale sightings in Australia")

grid = load()
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=6)
sub = grid[grid["month"] == month]

st.write(len(sub), "grid cells with sightings")
st_folium(make_map(grid, month), width=700, returned_objects=[])

st.subheader("Top 10 grid cells")
st.dataframe(sub.sort_values("n", ascending=False).head(10))

st.caption(
    "Circle size = number of sighting records, "
    "not encounter probability. Records cluster where "
    "observers are. Source: Atlas of Living Australia."
)
