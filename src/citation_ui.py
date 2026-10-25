import re
import streamlit as st
def display_answer(answer: str,msg_idx: int = 0):
    """把 [Page X] 渲染成可点击的 URL 链接。"""
    parts = re.split(r'(\[Page\s+\d+\])', answer)
    for part in parts:
        m = re.match(r'\[Page\s+(\d+)\]', part)
        if m:
            page = int(m.group(1))
            #  用 st.html 渲染真正的 <a> 链接
            st.html(
                f'<a href="?target_page={page}" target="_self" '
                f'style="display:inline-block;padding:4px 10px;margin:4px;'
                f'border:1px solid #ccc;border-radius:6px;'
                f'text-decoration:none;color:#1f77b4;font-size:14px;">'
                f'📖 Page {page}</a>'
            )#渲染真正的超链接
        else:
            if part.strip():
                st.markdown(part) #普通文本照常渲染
def handle_query_params():
    """读取 URL 里的 target_page，设置 session_state。"""
    if "target_page" in st.query_params:
        try:
            st.session_state.target_page = int(st.query_params["target_page"])
        except (ValueError, TypeError):
            pass
#整个流程：用户提问：“多头注意力是什么？”（见 app.py）1.大模型回答：“多头注意力是…… [Page 12]” 2.前端渲染：display_answer 把 [Page 12] 变成了一个 URL 为 ?target_page=12 的按钮。3.用户点击按钮：浏览器 URL 变成 localhost:8501/?target_page=12，触发页面重新加载。4.页面加载：app.py 第一步调用 handle_query_params()，把 12 存进 st.session_state.target_page。5.PDF 滚动：app.py 右侧的 pdf_viewer 读取 st.session_state.target_page，发现是 12，于是调用 scroll_to_page=12，PDF 自动翻到第 12 页。