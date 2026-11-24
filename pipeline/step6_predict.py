"""v1b→v2: マス目×月ごとの予測を作り、アプリ用に保存する。

テストのAUCが最も高いモデルを選ぶ(ベースラインに負けたら警告する)。
選んだモデルと同じ設定で、2000年〜昨年の全部の年を学習し直してから予測する(2026-11-24 追加)。
    テストの成績は step4・5 で、2019年までで学習したモデルを使って測ったもの。
    設定と作り方を確かめたあとに、最近の年の傾向も取り込むため、全部の年で学習し直す。
予測には、水温の平年値(sst_clim.csv)を使う。
予測するのは、2000年〜昨年に一度でも記録があるマスだけ(2026-11-24 修正)。
    それ以外のマス(2000年より前か、今年にしか記録がないマス)は、学習していないか、
    アプリで「この時期に見られた年」を示せないため、予測しない。

使い方:
    python pipeline/step6_predict.py          # 自動で選ぶ
    python pipeline/step6_predict.py logit    # モデルを指定する
    (このあと必ず step7_snap_to_sea.py を実行する)

出力:
    models/<モデル名>_full.joblib  全部の年で学習し直したモデル
    data/cell_points.csv glat, glon, plat, plon(マス目の中の実際の記録の平均位置。地図に印を置く場所)
    data/presence.csv    glat, glon, year, month(記録があったマス・年・月。アプリの「過去の実績」に使う)
    data/pred_month.csv  glat, glon, month, p, index, n_records
        p      : モデルの出力(相対的な見やすさ。遭遇確率そのものではない)
        index  : 全体の最大を100とした指数(アプリの表示用)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, FEATURES, MODELS, REPORTS, YEAR_MIN
from features import add_features  # noqa: E402  (同じフォルダ)


def choose_model(metrics: dict, forced: str | None) -> str:
    if forced:
        return forced
    cands = {k: metrics[f"auc_{k}"] for k in ("logit", "lgbm") if f"auc_{k}" in metrics}
    best = max(cands, key=cands.get)
    if cands[best] <= metrics["auc_baseline"]:
        print("警告: どのモデルもベースラインに勝っていません。設計を見直してください。")
    return best


def main(forced: str | None = None) -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    name = choose_model(metrics, forced)
    tested = joblib.load(MODELS / f"{name}.joblib")
    print("使うモデル:", name)

    # 同じ設定で、全部の年(2000年〜昨年。step3 で作った dataset.csv の全行)を学習し直す
    data = pd.read_csv(DATA / "dataset.csv")
    model = clone(tested).fit(data[FEATURES], data["y"])
    joblib.dump(model, MODELS / f"{name}_full.joblib")
    print("全部の年で学習し直した:", int(data["year"].min()), "〜", int(data["year"].max()), "年",
          len(data), "行 ->", MODELS / f"{name}_full.joblib")

    # 過去の実績: どのマスで、どの年のどの月に記録があったか。
    # いちばん新しい年は途中で、記録の登録も追いついていないので除く(全部そろった年だけを使う)
    s = pd.read_csv(DATA / "sightings.csv")
    last_full = int(s["year"].max()) - 1
    pres = (s[s["year"].between(YEAR_MIN, last_full)]
              [["glat", "glon", "year", "month"]].drop_duplicates())

    clim = pd.read_csv(DATA / "sst_clim.csv").dropna(subset=["sst"])
    # 予測するのは、2000年〜昨年に一度でも記録があるマスだけ。
    # それ以外のマスは学習していない(または実績を示せない)ので、予測が当てにならない
    n_all = clim[["glat", "glon"]].drop_duplicates().shape[0]
    clim = clim.merge(pres[["glat", "glon"]].drop_duplicates(), on=["glat", "glon"])
    n_use = clim[["glat", "glon"]].drop_duplicates().shape[0]
    print("予測するマス", n_all, "->", n_use, f"({YEAR_MIN}〜{last_full}年に記録がない {n_all - n_use} マスを除く)")

    grid = add_features(clim)
    grid["p"] = model.predict_proba(grid[FEATURES])[:, 1]
    grid["index"] = (grid["p"] / grid["p"].max() * 100).round(1)

    counts = pd.read_csv(DATA / "grid_month.csv").rename(columns={"n": "n_records"})
    grid = grid.merge(counts, on=["glat", "glon", "month"], how="left")
    grid["n_records"] = grid["n_records"].fillna(0).astype(int)

    out = grid[["glat", "glon", "month", "p", "index", "n_records"]]
    out.to_csv(DATA / "pred_month.csv", index=False)
    pres.to_csv(DATA / "presence.csv", index=False)

    # 地図の印は、マス目の中心ではなく、実際の記録の平均位置に置く(中心だと陸の上に出ることがあるため)
    pts = (s.groupby(["glat", "glon"])[["lat", "lon"]].mean().round(4)
             .rename(columns={"lat": "plat", "lon": "plon"}).reset_index())
    pts.to_csv(DATA / "cell_points.csv", index=False)
    print("保存:", DATA / "cell_points.csv", len(pts), "行(このあと step7 で海へ寄せる)")
    print("保存:", DATA / "presence.csv", len(pres), "行", f"({YEAR_MIN}〜{last_full}年)")
    print("保存:", DATA / "pred_month.csv", len(out), "行")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
