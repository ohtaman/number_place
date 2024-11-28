import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum, LpInteger
import numpy as np

# 数独の問題を解く関数
def solve_sudoku(grid):
    # 問題の定義
    prob = LpProblem("SudokuSolver", LpMinimize)

    # 変数の定義
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpInteger, lowBound=0, upBound=1)

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

    # 初期値の設定
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                prob += cells[i][j][grid[i][j]] == 1

    # 問題を解く
    prob.solve()

    # 解の取得
    if LpStatus[prob.status] == "Optimal":
        solved_grid = [[0 for _ in range(9)] for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() == 1:
                        solved_grid[i][j] = k
        return solved_grid
    else:
        return None


st.markdown("### 数理最適化（混合整数計画法）を使用して数独を解く")

# 数独の入力
st.write("問題を入力してください（0は空白を意味します）。")
hints = st.data_editor(np.zeros((9, 9)), hide_index=False)

if st.button("解く"):
    st.write("解を計算中...")
    solution = solve_sudoku(hints)
    if solution:
        st.write("解けました！")
        st.table(solution)
    else:
        st.write("解がありません。問題を確認してください。")
