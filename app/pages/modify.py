import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, LpStatus, LpBinary, lpSum
import numpy as np


def modify(hints):
    """
    ナンプレの初期状態を修正し、解を得る

    ユーザーが入力したナンプレの問題に対し、数理最適化を用いて解を導き出します。
    初期状態が矛盾している場合には、最小限の修正を加えた問題を生成します。

    Args:
        hints (np.ndarray): 9x9のナンプレの初期状態。NaNは空白を表す。

    Returns:
        tuple: (modified_hints, corrections, solution)
            - modified_hints (np.ndarray): 修正後の9x9ナンプレ問題。
            - corrections (list of tuple): 修正が加えられたセルの座標 [(i, j), ...]。
            - solution (np.ndarray): 修正後のナンプレ問題の解。
    """
    # 数理最適化問題の定義
    prob = LpProblem("ModifyNumberPlace", LpMinimize)

    # 各セルを表す変数を定義
    # cells[i][j][k] == 1 のとき、セル [i, j] の値が k であることを表す
    cells = LpVariable.dicts("Cell", (range(9), range(9), range(1, 10)), cat=LpBinary)
    # ペナルティ変数の定義
    # penalty[i][j] == 1 のとき、セル [i, j] の値（ヒント）を修正することを表す
    penalty = LpVariable.dicts("Penalty", (range(9), range(9)), cat=LpBinary)

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
                    cells[i][j][int(hints[i][j])] + penalty[i][j] == 1,
                    f"HintValue_{i}_{j}"
                )

    # 目的関数: ペナルティの合計を最小化
    prob += lpSum(penalty[i][j] for i in range(9) for j in range(9))

    # 解を見つける
    prob.solve()

    # 最適解が得られた場合、解を9x9グリッドで返す
    if LpStatus[prob.status] == "Optimal":
        # 解
        solution = np.zeros((9, 9))
        for i in range(9):
            for j in range(9):
                for k in range(1, 10):
                    if cells[i][j][k].value() == 1:
                        solution[i, j] = k
        # 修正箇所
        corrections = []
        for i in range(9):
            for j in range(9):
                if penalty[i][j].value() == 1:
                    corrections.append((i, j))

        # 修正後のヒント
        modified_hints = hints.copy()
        for i, j in corrections:
            modified_hints[i, j] = solution[i, j]

        return modified_hints, corrections, solution
    else:
        return None, None, None # 何らかの原因で解を得られなかった場合


def run():
    # Streamlitアプリ
    st.markdown("""
    ### 数理最適化で問題を修正する

    与えられたヒントに基づいて、数理最適化を用いてナンプレの問題を修正します。ここでは、既存のヒントが矛盾していて解が存在しない場合、最小限の修正で問題を解けるようにします。
    """)

    st.write("問題を入力してください。")
    hints = st.data_editor(np.full((9, 9), np.nan), hide_index=False, use_container_width=True)

    if st.button("修正する"):
        with st.spinner("解を計算中..."):
            modified_hints, corrections, solution = modify(hints)

        # 修正が必要な場合
        if corrections:
            st.subheader("修正後のヒント:")
            st.dataframe(modified_hints, hide_index=False)
            st.text(f"修正箇所: {corrections}")
        else:
            st.text(f"修正は必要ありません")

        if solution is not None:
            st.subheader("解:")
            st.dataframe(solution, hide_index=False)
        else:
            st.error("解がありません。問題を確認してください。")
