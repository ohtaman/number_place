import streamlit as st

pg = st.navigation([
    st.Page("pages/solve_problem.py", title="数独を解く"),
    st.Page("pages/generate_problem.py", title="数独を生成する"),
])
pg.run()