import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, LpBinary, lpSum
import numpy as np


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


# Streamlitアプリ
st.markdown("### 数理最適化でナンプレ（ナンプレ）を解く")

st.write("問題を入力してください。")
hints = st.data_editor(np.full((9, 9), np.nan), hide_index=False, use_container_width=True)

if st.button("解く"):
    with st.spinner("解を計算中..."):
        solution = solve(hints)

    # 解が得られた場合、結果を表示
    if solution is not None:
        st.subheader("解:")
        st.dataframe(solution, hide_index=False)
    else:
        st.error("解がありません。問題を確認してください。")
