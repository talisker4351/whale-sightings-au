"""v1b: LightGBMの設定を調整してから学習し、ベースラインとロジスティック回帰と比べる。

設定の選び方(2026-11-24 追加):
    テスト用の年(2020年以降)は、設定選びには一切使わない。
    学習用の年(2000〜2019年)をさらに分け、2000〜2014年で学習・2015〜2019年で確かめて、
    いくつかの設定の中から、確かめ用の年のAUCが最も高いものを選ぶ。
    木の本数は「早期打ち切り」(確かめ用のAUCが50本続けて上がらなければ止める)で決める。
    選んだ設定で2000〜2019年を学習し直し、2020年以降のテストで一度だけ成績を測る。
    比較のため、調整前の設定(これまでの値)のテスト成績も記録する。

使い方(step4 のあとに実行):
    python pipeline/step5_train_lgbm.py

出力:
    models/lgbm.joblib
    reports/metrics.json に auc_lgbm、auc_lgbm_default、lgbm_params、tuning などを追記
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, FEATURES, MODELS, REPORTS, TEST_FROM
from step4_train_logit import calib_table, split  # noqa: E402  (同じフォルダ)

VALID_YEARS = 5          # 学習用の年のうち、最後の5年(2015〜2019年)を確かめ用にする
COMMON = dict(learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)
DEFAULT = dict(n_estimators=400, num_leaves=31, min_child_samples=50)   # 調整前の設定
GRID = {                                                                # 試す設定
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 50, 200],
}
MAX_TREES = 2000

# 新しい版の LightGBM では eval_set に「将来使えなくなる」という注意が出るが、今は問題なく動くので表示しない
warnings.filterwarnings("ignore", message=".*eval_set.*")


def tune(train: pd.DataFrame) -> tuple[dict, list]:
    """学習用の年の中だけで設定を選ぶ。テスト用の年は見ない。"""
    valid_from = TEST_FROM - VALID_YEARS
    fit, valid = train[train["year"] < valid_from], train[train["year"] >= valid_from]
    print(f"設定選び: {int(fit['year'].min())}〜{valid_from - 1}年で学習 / "
          f"{valid_from}〜{TEST_FROM - 1}年で確かめる")
    results = []
    for leaves, child in itertools.product(GRID["num_leaves"], GRID["min_child_samples"]):
        m = LGBMClassifier(n_estimators=MAX_TREES, num_leaves=leaves, min_child_samples=child, **COMMON)
        m.fit(fit[FEATURES], fit["y"], eval_set=[(valid[FEATURES], valid["y"])], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        trees = int(m.best_iteration_ or MAX_TREES)
        auc = roc_auc_score(valid["y"], m.predict_proba(valid[FEATURES], num_iteration=trees)[:, 1])
        results.append({"num_leaves": leaves, "min_child_samples": child,
                        "n_estimators": trees, "auc_valid": round(float(auc), 4)})
        print(f"  葉{leaves:>3} 最低行数{child:>4} 木{trees:>5}本  確かめ用AUC {auc:.4f}")
    best = max(results, key=lambda r: r["auc_valid"])
    params = {k: best[k] for k in ("num_leaves", "min_child_samples", "n_estimators")}
    print("選んだ設定:", params)
    return params, results


def main() -> None:
    df = pd.read_csv(DATA / "dataset.csv")
    train, test = split(df)

    params, results = tune(train)

    # 選んだ設定で、学習用の年(2000〜2019年)全部を学習し直し、テストで一度だけ測る
    model = LGBMClassifier(**params, **COMMON)
    model.fit(train[FEATURES], train["y"])
    p = model.predict_proba(test[FEATURES])[:, 1]

    # 比較用: 調整前の設定
    base = LGBMClassifier(**DEFAULT, **COMMON).fit(train[FEATURES], train["y"])
    p0 = base.predict_proba(test[FEATURES])[:, 1]

    metrics = json.loads((REPORTS / "metrics.json").read_text())
    metrics["auc_lgbm"] = round(roc_auc_score(test["y"], p), 4)
    metrics["auc_lgbm_default"] = round(roc_auc_score(test["y"], p0), 4)
    metrics["lgbm_params"] = {**params, **{k: v for k, v in COMMON.items() if k != "verbose"}}
    metrics["tuning"] = results
    metrics["calibration_lgbm"] = calib_table(test["y"], p)
    metrics["importance_lgbm"] = dict(zip(FEATURES, [int(v) for v in model.feature_importances_]))

    print("AUC ベースライン          ", metrics["auc_baseline"])
    print("AUC ロジスティック回帰    ", metrics["auc_logit"])
    print("AUC LightGBM(調整前)    ", metrics["auc_lgbm_default"])
    print("AUC LightGBM(調整後)    ", metrics["auc_lgbm"])

    joblib.dump(model, MODELS / "lgbm.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("保存:", MODELS / "lgbm.joblib", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
