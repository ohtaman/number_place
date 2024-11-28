import streamlit as st

pages = st.navigation([
    st.Page("pages/solve.py", title="数独を解く"),
    st.Page("pages/modify.py", title="問題を修正する"),
    st.Page("pages/generate.py", title="問題を生成する"),
    st.Page("pages/generate_random.py", title="ランダムな問題を生成する"),
    st.Page("pages/generate_with_hints.py", title="指定したセルを固定して生成"),
])

pages.run()