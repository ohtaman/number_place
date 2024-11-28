import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum, LpInteger
import pandas as pd
import numpy as np

# 数独を解き、誤りを修正する関数
def solve_and_correct_sudoku_with_penalty(grid):
    # 問題の定義
    prob = LpProblem("SudokuSolver", LpMinimize)

    # 変数の定義
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpInteger, lowBound=0, upBound=1)

    # Penalty変数の定義（修正が必要なセルを表す）
    penalty = LpVariable.dicts("Penalty", (range(9), range(9)), cat=LpInteger, lowBound=0, upBound=1)

    # 制約条件
    for i in range(9):
        for j in range(9):
            prob += lpSum(cells[i][j][k] for k in range(1, 10)) == 1

    for i in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for j in range(9)) == 1

    for j in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for i in range(9)) == 1

    for block_i in range(3):
        for block_j in range(3):
            for k in range(1, 10):
                prob += lpSum(
                    cells[block_i * 3 + di][block_j * 3 + dj][k] for di in range(3) for dj in range(3)
                ) == 1

    # 初期値の設定とPenalty変数の導入
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                k = grid[i][j]
                # 入力値を尊重しつつ、修正を許容（Penaltyで修正数を最小化）
                prob += cells[i][j][k] + penalty[i][j] == 1

    # 目的関数: Penaltyの合計を最小化
    prob += lpSum(penalty[i][j] for i in range(9) for j in range(9))

    # 問題を解く
    prob.solve()

    # 解の取得
    if LpStatus[prob.status] == "Optimal":
        solved_grid = [[0 for _ in range(9)] for _ in range(9)]
        corrections = []
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() > 0.5:
                        solved_grid[i][j] = k
                if penalty[i][j].value() > 0.5:
                    corrections.append((i + 1, j + 1))  # 1-indexedで返す
        return solved_grid, corrections
    else:
        return None, None


# Streamlit アプリ
st.title("数独ソルバー (修正提案付き)")
st.write("混合整数計画法を使用して数独を解き、誤りを修正します。")

# 数独の入力
st.write("問題を入力してください（0は空白を意味します）。")
hints = st.data_editor(np.zeros((9, 9), dtype=int), hide_index=False)

if st.button("解く"):
    st.write("解を計算中...")
    solution, corrections = solve_and_correct_sudoku_with_penalty(hints)
    if solution:
        st.write("解けました！")
        st.subheader("修正された問題:")
        corrected_grid = hints.copy()
        for i, j in corrections:
            corrected_grid[i - 1][j - 1] = 0  # 修正箇所を空白に
        st.table(pd.DataFrame(corrected_grid, index=range(1, 10), columns=range(1, 10)))

        st.subheader("修正箇所:")
        st.write(f"{len(corrections)} 箇所の修正が必要です: {corrections}")

        st.subheader("解:")
        st.table(pd.DataFrame(solution, index=range(1, 10), columns=range(1, 10)))
    else:
        st.write("解けませんでした。問題を確認してください。")
