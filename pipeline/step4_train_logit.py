"""v1a: ロジスティック回帰で学習し、ベースラインと比べる。

年で区切って評価する(TEST_FROM 年より前で学び、それ以降で確かめる)。

使い方:
    python pipeline/step4_train_logit.py

出力:
    models/logit.joblib
    reports/metrics.json   (baseline と logit のAUCなど)
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, FEATURES, MODELS, REPORTS, TEST_FROM


def split(df):
    return df[df["year"] < TEST_FROM], df[df["year"] >= TEST_FROM]


def baseline_scores(train, test):
    """ベースライン: 学習期間の「マス×月で記録があった年の割合」をそのまま予測に使う。"""
    base = train.groupby(["glat", "glon", "month"])["y"].mean().rename("base")
    return test.join(base, on=["glat", "glon", "month"])["base"].fillna(0)


def calib_table(y, p, bins=5):
    frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return [{"predicted": round(float(a), 4), "actual": round(float(b), 4)}
            for a, b in zip(mean_p, frac)]


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    train, test = split(df)
    print("学習", len(train), "行 / テスト", len(test), "行")

    # 水温と緯度は単位が違うので、そろえてから学習する(係数を比べやすくなる)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[FEATURES], train["y"])

    p = model.predict_proba(test[FEATURES])[:, 1]
    b = baseline_scores(train, test)
    metrics = {
        "auc_baseline": round(roc_auc_score(test["y"], b), 4),
        "auc_logit": round(roc_auc_score(test["y"], p), 4),
        "calibration_logit": calib_table(test["y"], p),
        "coef_logit_standardized": dict(zip(
            FEATURES, [round(float(c), 4) for c in model[-1].coef_[0]])),
        "test_from": TEST_FROM,
    }
    print("AUC ベースライン", metrics["auc_baseline"])
    print("AUC ロジスティック回帰", metrics["auc_logit"])
    print("係数(標準化後。1標準偏差あたりの効き方)", metrics["coef_logit_standardized"])

    joblib.dump(model, MODELS / "logit.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("保存:", MODELS / "logit.joblib", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
