"""共通の設定。数字を変えたいときは、ここだけを直す。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # 生成したCSV(公開に必要なものだけGitHubに載せる)
RAW = DATA / "raw"            # ALAからダウンロードしたファイル(GitHubには載せない)
MODELS = ROOT / "models"      # 学習済みモデル
REPORTS = ROOT / "reports"    # 評価結果

SPECIES = "Megaptera novaeangliae"   # ザトウクジラ
STEP = 0.25                          # マス目の刻み(度)。約25km
LAT_MIN, LAT_MAX = -45, -9           # オーストラリア周辺の範囲
LON_MIN, LON_MAX = 110, 156

YEAR_MIN = 2000        # 学習に使う最初の年(古い年は記録が少ない)
TEST_FROM = 2020       # この年以降をテストに使う(年で区切る)

# 海面水温: NOAA OISST v2 高解像度(0.25度)の月平均。OPeNDAPで必要な範囲だけ読む
SST_URL = ("https://psl.noaa.gov/thredds/dodsC/"
           "Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc")

FEATURES = ["glat", "glon", "msin", "mcos", "lat_msin", "lat_mcos", "sst"]

for d in (DATA, RAW, MODELS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)
