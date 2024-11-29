import random

import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, LpBinary, lpSum
import numpy as np
import pandas as pd


# ナンプレを解く関数
def solve(hints):
    """
    ナンプレを混合整数計画法で解く

    Args:
        hints (np.ndarray): 9x9のナンプレの初期状態。NaNは空白を表す。

    Returns:
        list: 解の9x9グリッド（解がない場合はNone）。
    """
    # 数理最適化問題の定義
    prob = LpProblem("SolveNumberPlace", LpMinimize)

    # 各セルを表す変数を定義
    # cells[i][j][k] == 1 のとき、セル [i, j] の値が k であることを表す
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpBinary)

    # 各セルに1つの値が入る制約
    for i in range(9):
        for j in range(9):
            prob.addConstraint(
                lpSum(cells[i][j][k] for k in range(1, 10)) == 1,
                f"CellValue_{i}_{j}"
            )

    # 各行に1〜9が1回ずつ入る制約
    for i in range(9):
        for k in range(1, 10):
            prob.addConstraint(
                lpSum(cells[i][j][k] for j in range(9)) == 1,
                f"RowValue_{i}_{k}"
            )

    # 各列に1〜9が1回ずつ入る制約
    for j in range(9):
        for k in range(1, 10):
            prob.addConstraint(
                lpSum(cells[i][j][k] for i in range(9)) == 1,
                f"ColumnValue_{j}_{k}"
            )

    # 各3x3ブロックに1〜9が1回ずつ入る制約
    for block_i in range(3):
        for block_j in range(3):
            for k in range(1, 10):
                prob.addConstraint(
                    lpSum(cells[block_i * 3 + di][block_j * 3 + dj][k] for di in range(3) for dj in range(3)) == 1,
                    f"BlockValue_{block_i}_{block_j}_{k}",
                )

    # 初期値（ヒント）を設定
    for i in range(9):
        for j in range(9):
            if not np.isnan(hints[i][j]):  # NaNは空白を表す
                prob.addConstraint(
                    cells[i][j][int(hints[i][j])] == 1,
                    f"HintValue_{i}_{j}"
                )

    # 目的関数を設定（必要ないため定数（0）を設定）
    prob.setObjective(lpSum([]))

    # 解を見つける
    prob.solve()

    # 最適解が得られた場合、解を9x9グリッドで返す
    if LpStatus[prob.status] == "Optimal":
        solution = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() == 1:
                        solution[i, j] = k
        return solution
    else:
        return None  # 解がない場合


def has_unique_solution(hints):
    """
    ナンプレに一意解が存在するか確認する関数。

    Args:
        hints (np.ndarray): 9x9のナンプレの初期状態。NaNは空白を表す。

    Returns:
        bool: 一意解が存在すればTrue、そうでなければFalse。
    """
    # 数理最適化問題の定義
    prob = LpProblem("SolveNumberPlace", LpMinimize)

    # 各セルを表す変数を定義
    # cells[i][j][k] == 1 のとき、セル [i, j] の値が k であることを表す
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpBinary)

    # 各セルに1つの値が入る制約
    for i in range(9):
        for j in range(9):
            prob.addConstraint(
                lpSum(cells[i][j][k] for k in range(1, 10)) == 1,
                f"CellValue_{i}_{j}"
            )

    # 各行に1〜9が1回ずつ入る制約
    for i in range(9):
        for k in range(1, 10):
            prob.addConstraint(
                lpSum(cells[i][j][k] for j in range(9)) == 1,
                f"RowValue_{i}_{k}"
            )

    # 各列に1〜9が1回ずつ入る制約
    for j in range(9):
        for k in range(1, 10):
            prob.addConstraint(
                lpSum(cells[i][j][k] for i in range(9)) == 1,
                f"ColumnValue_{j}_{k}"
            )

    # 各3x3ブロックに1〜9が1回ずつ入る制約
    for block_i in range(3):
        for block_j in range(3):
            for k in range(1, 10):
                prob.addConstraint(
                    lpSum(cells[block_i * 3 + di][block_j * 3 + dj][k] for di in range(3) for dj in range(3)) == 1,
                    f"BlockValue_{block_i}_{block_j}_{k}",
                )

    # 初期値（ヒント）を設定
    for i in range(9):
        for j in range(9):
            if not np.isnan(hints[i][j]):  # NaNは空白を表す
                prob.addConstraint(
                    cells[i][j][int(hints[i][j])] == 1,
                    f"HintValue_{i}_{j}"
                )

    # 目的関数を設定（必要ないため定数（0）を設定）
    prob.setObjective(lpSum([]))

    # 解を見つける
    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        return False  # 解が存在しない


    # はじめに見つかった解を禁止する
    # はじめに見つかった解で選択されなかった変数 cells[i][j][k] が少なくとも1つは値 1 を持つ
    prob.addConstraint(
        lpSum(
            cells[i][j][k]
            for i in range(9)
            for j in range(9)
            for k in range(1, 10)
            if cells[i][j][k].value() != 1
        ) >= 1
    )

    prob.solve()

    return LpStatus[prob.status] != "Optimal"  # 別解がなければ一意解


def generate_problem(solution, num_hints):
    """
    一意解を持つナンプレの問題を生成する

    Args:
        solution (np.ndarray): 完全なナンプレの解。
        num_hints (int): 問題として残すセルの数（難易度調整）。

    Returns:
        np.ndarray: 一意解を持つナンプレの問題。
    """
    hints = solution.copy()
    indices = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(indices)  # セルの順序をランダムにする

    for i, j in indices:
        # 現在のセルを削除（np.nan に設定）
        temp = hints[i][j]
        hints[i][j] = np.nan

        # 一意解でなければ削除を戻す
        if not has_unique_solution(hints):
            hints[i][j] = temp

        # 残っているヒントの数が指定の数に達したら終了
        if np.sum(~np.isnan(hints)) == num_hints:
            break

    return hints


# Streamlitアプリ
st.markdown("### 数理最適化で問題を生成する")

# 難易度の選択（ヒントの数を指定）
num_hints = st.slider("ヒントの数（目標値）を選択してください", min_value=17, max_value=81, value=30)


if st.button("生成する"):
    with st.spinner("解を生成中..."):
        solution = solve(np.full((9, 9), np.nan))
    if solution is not None:
        with st.spinner("問題を生成中..."):
            hints = generate_problem(solution, num_hints)

        st.subheader("生成された問題:")
        st.dataframe(hints)

        st.subheader("解:")
        st.dataframe(solution)
    else:
        st.write("解を生成できませんでした。")
