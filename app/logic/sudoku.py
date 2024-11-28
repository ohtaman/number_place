from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, lpSum, LpBinary
import numpy as np

def solve(grid: np.ndarray) -> np.ndarray:
    """
    数独の解を求める。

    Args:
        grid (np.ndarray): 9x9 の数独の初期配置。未定のセルは 0。

    Returns:
        np.ndarray: 解かれた 9x9 の数独グリッド。解が存在しない場合は None。
    """
    prob = LpProblem("SudokuSolver", LpMinimize)

    # セル [i, j] の値が k のとき cells[i][j][k] == 1
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpBinary)

    # 制約1: 各セルには1つの値のみが入る
    for i in range(9):
        for j in range(9):
            prob += lpSum(cells[i][j][k] for k in range(1, 10)) == 1

    # 制約2: 各行には各数字が1回だけ入る
    for i in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for j in range(9)) == 1

    # 制約3: 各列には各数字が1回だけ入る
    for j in range(9):
        for k in range(1, 10):
            prob += lpSum(cells[i][j][k] for i in range(9)) == 1

    # 制約4: 各3x3ブロックには各数字が1回だけ入る
    for block_i in range(3):
        for block_j in range(3):
            for k in range(1, 10):
                prob += lpSum(
                    cells[block_i * 3 + di][block_j * 3 + dj][k] for di in range(3) for dj in range(3)
                ) == 1

    # 制約5: 初期値の設定（入力グリッドのヒントを反映）
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                prob += cells[i][j][grid[i][j]] == 1

    # 数独を解く
    prob.solve()

    # 解の取得
    if LpStatus[prob.status] == "Optimal":
        solution = np.zeros((9, 9), dtype=int)
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() == 1:
                        solution[i, j] = k
        return solution
    else:
        return None
