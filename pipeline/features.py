"""学習と予測で同じ特徴量を作るための関数(step3 と step6 が使う)。"""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """月を円周上の位置に変え、緯度との組み合わせを足す。

    sin と cos で表すと、12月と1月が近いことをモデルに伝えられる。
    緯度との組み合わせは、南北で季節の山がずれる回遊を表すため。
    """
    df = df.copy()
    angle = 2 * np.pi * df["month"] / 12
    df["msin"] = np.sin(angle)
    df["mcos"] = np.cos(angle)
    df["lat_msin"] = df["glat"] * df["msin"]
    df["lat_mcos"] = df["glat"] * df["mcos"]
    return df
