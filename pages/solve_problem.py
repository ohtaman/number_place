import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum, LpInteger
import pandas as pd
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

    # 最小化の目的関数（必要なしだがダミーで0を設定）
    prob += 0

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


# Streamlit アプリ
st.title("数独ソルバー")
st.write("混合整数計画法を使用して数独を解きます。")

# 数独の入力
st.write("数独の問題を入力してください（空白は \"\" または 0 を意味します）。")

# デフォルトの空白グリッドを作成
options = [""]
options.extend([str(i) for i in range(1, 10)])

# インデックスを1から始める
default_grid = pd.DataFrame(
    "", 
    index=pd.Index(range(1, 10), name="行"),  # 行のインデックスを1から9に
    columns=pd.Index(range(1, 10), name="列")  # 列のインデックスを1から9に
)

# 各セルにセレクターを設定
grid_df = st.data_editor(
    default_grid.astype("category").apply(lambda col: col.cat.set_categories(options)),
    hide_index=False,
    use_container_width=True,
    key="sudoku_grid",
)

# DataFrame を数値に変換（空白は 0 に置き換え）
grid = grid_df.replace("", 0).astype(int).values

if st.button("解く"):
    st.write("解を計算中...")
    solution = solve_sudoku(grid)
    if solution:
        st.write("解けました！")
        st.table(pd.DataFrame(solution, index=range(1, 10), columns=range(1, 10)))
    else:
        st.write("解けませんでした。問題を確認してください。")
