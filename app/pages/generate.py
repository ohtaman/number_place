import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpStatus
import numpy as np
import pandas as pd
import random

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


# 数独が一意解を持つか確認する関数
def is_unique_solution(grid):
    # モデルを再定義
    prob = LpProblem("SudokuUniquenessChecker", LpMinimize)

    # 変数の定義
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpInteger, lowBound=0, upBound=1)

    # 制約条件
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                prob += cells[i][j][grid[i][j]] == 1
            else:
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
                    cells[block_i * 3 + di][block_j * 3 + dj][k]
                    for di in range(3) for dj in range(3)
                ) == 1

    # 最初の解を求める
    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        return False  # 解が存在しない

    first_solution = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            for k in range(1, 10):
                if cells[i][j][k].value() > 0.5:
                    first_solution[i][j] = k

    # 制約に新しい条件を追加し、別の解が存在するか確認
    prob += lpSum(
        cells[i][j][k] * (1 if first_solution[i][j] != k else 0)
        for i in range(9) for j in range(9) for k in range(1, 10)
    ) >= 1
    prob.solve()

    return LpStatus[prob.status] != "Optimal"  # 別解がなければ一意解

# 一意解を持つ数独問題を生成する関数
def generate_sudoku_problem(solution):
    problem = [row[:] for row in solution]
    indices = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(indices)

    for i, j in indices:
        temp = problem[i][j]
        problem[i][j] = 0
        if not is_unique_solution(problem):
            problem[i][j] = temp  # 元に戻す
    return problem

# Streamlit アプリ
st.title("数独問題生成アプリ (数理最適化使用)")
st.write("一意解を持つ数独問題を生成します。")

# 問題を生成
if st.button("数独問題を生成"):
    solution = solve_sudoku(np.zeros((9,9), dtype=int))
    if solution:
        with st.spinner("問題を作成中..."):
            problem = generate_sudoku_problem(solution)

        st.subheader("生成された問題:")
        st.table(pd.DataFrame(problem))

        st.subheader("解:")
        st.table(pd.DataFrame(solution))
    else:
        st.write("数独の解を生成できませんでした。")