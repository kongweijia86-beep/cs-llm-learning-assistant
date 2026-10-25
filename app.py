import os
import uuid
import shutil
from pathlib import Path
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from langchain_core.messages import HumanMessage
from src.rag import RAG
from src.agent_graph import build_agent
from src.citation_ui import display_answer,handle_query_params
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR/'data'/'raw'
DEFAULT_PDF = PDF_DIR/"Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf"
def render_contexts(contexts):
    if not contexts:
        return []
    with st.expander('参考资料',expanded=True):
        for item in contexts:
            st.write(f"**{item['source']}-Page{item['page']}**(score={item.get('score',0.0):.3f})" )
            st.caption(item['text'][:200]+'...')
def main():
    st.set_page_config(page_title='NLP学习助手',page_icon='🧠',layout="wide")
    handle_query_params()
    st.title('🧠大模型学习助手')
    st.caption('基于tool calling的agent(支持动态PDF上传)')
    with st.sidebar:
        st.header("知识库管理")
        uploaded_file = st.file_uploader('上传新的PDF文档',type=['pdf'])
        if uploaded_file is not None:
            save_path = PDF_DIR/uploaded_file.name
            if not save_path.exists():
                with open(save_path,'wb') as f:## 以二进制模式打开（或创建）这个文件
                    f.write(uploaded_file.getbuffer())
                    st.success(f"文件{uploaded_file.name}已经保存")
                    st.cache_resource.clear() #强制重新加载知识库
                    st.rerun()
        st.divider()
        st.header('设置')
        if st.button("新建会话(清空回忆)"):
            st.session_state.thread_id = str(uuid.uuid4()) #uuid.uuid4()：生成一个全球唯一的字符串
            st.session_state.chat_history = []
            st.rerun()
    @st.cache_resource
    def load_resources(pdf_dir_str,current_files_str):
        rag = RAG(pdf_dir_str)
        agent = build_agent(rag)
        return rag,agent
    with st.spinner("正在加载知识库并编译 Agent 图..."):
        current_files = sorted([f.name for f in PDF_DIR.glob('*pdf')])
        try:
            rag,agent = load_resources(str(PDF_DIR),str(current_files))
        except Exception as e:
            st.error(f"知识库加载失败，可能是上传的PDF无法被解析:{e}")
            st.stop()
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    col_chat,col_pdf = st.columns([1,1])
    with col_chat:
        for idx,msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg['role']):
                if msg['role'] == 'assistant':
                    display_answer(msg['content'],msg_idx=idx)
                else:
                    st.write(msg['content'])
                #渲染参考资料
                if msg['role'] == 'assistant' and 'contexts' in msg:
                    render_contexts(msg['contexts'])
    question = st.chat_input('请输入你的问题')
    if question:
        st.session_state.chat_history.append({'role':'user','content':question})
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            with st.spinner('Agent思考中'):
                config = {'configurable':{'thread_id':st.session_state.thread_id}}
                inputs = {
                    'messages':[HumanMessage(content=question)]
                }
                final_state = agent.invoke(inputs,config=config)
                answer = final_state.get('answer','未能生成答案')
                contexts = final_state.get('contexts',[])
            msg_idx = len(st.session_state.chat_history)
            display_answer(answer,msg_idx=msg_idx)
            render_contexts(contexts)
        st.session_state.chat_history.append({
            'role':'assistant',
            'content':answer,
            'contexts':contexts
        })
    with col_pdf:
        st.subheader('PDF预览') #st.subheader()是加一个小标题
        target_pdf = str(DEFAULT_PDF)
        if not DEFAULT_PDF.exists():
            pdf_files = list(PDF_DIR.glob("*.pdf"))
            if pdf_files:
                target_pdf = str(pdf_files[0])
            else:
                st.warning("没有找到任何 PDF 文件")
                return
        target_page = st.session_state.get('target_page',None)
        pdf_viewer(
            input = target_pdf,
            width = 600,
            height = 900,
            scroll_to_page = target_page,
            key = f"pdf_viewer:{target_page}" #Streamlit 要求每个组件有唯一的 Key。
        )

if __name__ == '__main__':
    main()









