"""v1b: LightGBMで学習し、ベースラインとロジスティック回帰と比べる。

使い方(step4 のあとに実行):
    python pipeline/step5_train_lgbm.py

出力:
    models/lgbm.joblib
    reports/metrics.json に auc_lgbm などを追記
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, FEATURES, MODELS, REPORTS
from step4_train_logit import calib_table, split  # noqa: E402  (同じフォルダ)


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    train, test = split(df)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,     # 葉に最低50行。細かく覚えすぎるのを防ぐ
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[FEATURES], train["y"])
    p = model.predict_proba(test[FEATURES])[:, 1]

    metrics = json.loads((REPORTS / "metrics.json").read_text())
    metrics["auc_lgbm"] = round(roc_auc_score(test["y"], p), 4)
    metrics["calibration_lgbm"] = calib_table(test["y"], p)
    metrics["importance_lgbm"] = dict(zip(
        FEATURES, [int(v) for v in model.feature_importances_]))

    print("AUC ベースライン      ", metrics["auc_baseline"])
    print("AUC ロジスティック回帰", metrics["auc_logit"])
    print("AUC LightGBM          ", metrics["auc_lgbm"])
    print("特徴量の重要度(分岐に使われた回数)", metrics["importance_lgbm"])

    joblib.dump(model, MODELS / "lgbm.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("保存:", MODELS / "lgbm.joblib", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
