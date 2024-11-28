import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpStatus
import numpy as np
import pandas as pd
import random

# 数独の解を生成する関数（重み付き）
def generate_full_sudoku_solution_with_weights():
    # 問題の定義
    prob = LpProblem("SudokuGeneratorWithWeights", LpMinimize)

    # ランダムな重みを生成
    weights = np.random.randint(1, 10, size=(9, 9))

    # 変数の定義
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpInteger, lowBound=0, upBound=1)

    # 制約条件
    # 各セルには1つの値
    for i in range(9):
        for j in range(9):
            prob += lpSum(cells[i][j][k] for k in range(1, 10)) == 1

    # 各行には1〜9が1回ずつ
    for i in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for j in range(9)) == 1

    # 各列には1〜9が1回ずつ
    for j in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for i in range(9)) == 1

    # 各3x3ブロックには1〜9が1回ずつ
    for block_i in range(3):
        for block_j in range(3):
            for k in range(1, 10):
                prob += lpSum(
                    cells[block_i * 3 + di][block_j * 3 + dj][k]
                    for di in range(3) for dj in range(3)
                ) == 1

    # 目的関数: ランダムな重みを最小化（解のランダム性を増やす）
    prob += lpSum(weights[i][j] * lpSum(cells[i][j][k] for k in range(1, 10)) for i in range(9) for j in range(9))

    # 問題を解く
    prob.solve()

    # 解を取得
    if LpStatus[prob.status] == "Optimal":
        solution = [[0 for _ in range(9)] for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() > 0.5:
                        solution[i][j] = k
        return solution, weights
    else:
        return None, None

# 数独が一意解を持つか確認する関数（変更なし）
def is_unique_solution(grid):
    prob = LpProblem("SudokuUniquenessChecker", LpMinimize)
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpInteger, lowBound=0, upBound=1)
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
    prob.solve()
    if LpStatus[prob.status] != "Optimal":
        return False
    first_solution = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            for k in range(1, 10):
                if cells[i][j][k].value() > 0.5:
                    first_solution[i][j] = k
    prob += lpSum(
        cells[i][j][k] * (1 if first_solution[i][j] != k else 0)
        for i in range(9) for j in range(9) for k in range(1, 10)
    ) >= 1
    prob.solve()
    return LpStatus[prob.status] != "Optimal"

# 一意解を持つ数独問題を生成する関数（変更なし）
def generate_sudoku_problem(solution):
    problem = [row[:] for row in solution]
    indices = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(indices)
    for i, j in indices:
        temp = problem[i][j]
        problem[i][j] = 0
        if not is_unique_solution(problem):
            problem[i][j] = temp
    return problem

# Streamlit アプリ
st.title("数独問題生成アプリ (重み付き数理最適化使用)")
st.write("一意解を持つ数独問題を生成します。")

# 問題を生成
if st.button("数独問題を生成"):
    st.write("数独の解を生成中...")
    solution, weights = generate_full_sudoku_solution_with_weights()
    if solution:
        st.write("数独の解が生成されました。問題を作成中...")
        problem = generate_sudoku_problem(solution)

        st.subheader("生成された問題:")
        st.table(pd.DataFrame(problem))

        st.subheader("解:")
        st.table(pd.DataFrame(solution))

        st.subheader("ランダムな重み行列:")
        st.table(pd.DataFrame(weights))
    else:
        st.write("数独の解を生成できませんでした。")
