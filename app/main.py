import streamlit as st

pages = st.navigation([
    st.Page("pages/solve.py", title="ナンプレを解く"),
    st.Page("pages/modify.py", title="問題を修正する"),
    st.Page("pages/generate.py", title="問題を生成する"),
    st.Page("pages/generate_random.py", title="多様な解の問題を生成する"),
    st.Page("pages/generate_with_hints.py", title="セルの値を固定して生成"),
])

pages.run()
with st.sidebar:
    st.markdown("[解説](https://zenn.dev/ohtaman/articles/opt-number-place)")