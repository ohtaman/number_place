import streamlit as st

from app.pages import generate_random, generate_with_hints, generate, modify, solve

pages = st.navigation([
    st.Page(solve.run, title="ナンプレを解く", url_path="solve"),
    st.Page(modify.run, title="問題を修正する", url_path="modify"),
    st.Page(generate.run, title="問題を生成する", url_path="generate"),
    st.Page(generate_random.run, title="多様な解の問題を生成する", url_path="generate_random"),
    st.Page(generate_with_hints.run, title="セルの値を固定して生成", url_path="generate_with_hints"),
])

pages.run()
with st.sidebar:
    st.markdown("[解説](https://zenn.dev/ohtaman/articles/opt-number-place)")