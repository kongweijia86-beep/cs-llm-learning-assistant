from typing import TypedDict,Annotated,List,Dict,Optional
import operator
import json
import sqlite3
from langgraph.graph import StateGraph,END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from src.rag import RAG
_rag_instance = None
#计算注意力机制的 FLOPs（算力成本），n 是句子长度（序列长度），dk是每个头的维度。
#“它是用来估算大模型算力开销的。因为注意力的计算量跟序列长度（n）是平方关系（n^2），所以一旦文本变长，算力成本会呈指数级暴增。这个工具能帮我们在调用大模型前，提前评估配置（比如头维度 dk和上下文长度）是否合理，避免显卡跑爆。”
def _calc_attention_complexity(d_k:int,seq_len:int,n_heads: int = 1, n_layers: int = 1)->dict:
    pre_head = 4 * (seq_len ** 2) * d_k +2*(seq_len**2)
    flops = pre_head * n_heads * n_layers
    return {
        "formula": "FLOPs ≈ (4 * n^2 * d_k + 2 * n^2) * n_heads * n_layers",
        "d_k": d_k, "seq_len": seq_len, "flops": flops,
        "flops_readable": f"{flops:.3e}",
        "note": "这是单头、单次前向的近似；多头需再乘 head 数。",
    }
#d_model（模型维度 / 隐藏层维度 / Hidden Size）.简单来说，它是模型内部所有特征向量的长度。
#计算注意力模块的参数量
def _calc_attention_params(d_model:int,n_heads:int)->dict:
    d_k = d_model//n_heads #每个头的维度
    #Linear layer(线性投影层)的参数量(权重矩阵+偏置)
    pre_proj = d_model * d_model + d_model
    total = 4 * pre_proj #乘以4因为在标准的多头注意力模块中，有 4 个 独立的线性投影层：Wq(query投影)吧输入变成query,Wk(key投影)把输入变成key,Wv(value投影)把输入变成value,Wo(output投影)把多头拼接后的结果混合输出
    #这里其实暗含了一个多头的特性——虽然我们拆成了 8 个头，但在代码底层实现时，矩阵乘法通常是把 8 个头拼在一起算的，相当于一个大的d_model*d_model矩阵，因此不管有几个头，投影层的总参数量始终是4*（d_model^2+d_model),与n_heads无关
    return {
        "formula": "params = 4 * (d_model^2 + d_model)",
        "d_model": d_model, "n_heads": n_heads, "d_k": d_k,
        "params": total, "note": "不含 LayerNorm 和 FFN。",
    }
#计算FFN的参数量
def _calc_ffn_params(d_model:int,expansion:int=4)->dict:
    d_ff = d_model * expansion #计算隐藏层维度d_ff
    #计算两层线性层的参数量
    params = (d_model*d_ff+d_ff) + (d_ff*d_model+d_model)
    return {
        "formula": "params = 2 * d_model * d_ff + d_ff + d_model",
        "d_model": d_model, "d_ff": d_ff, "expansion": expansion, "params": params,
    }

@tool
def calculate_attention_complexity(d_k:int,seq_len:int)->str:
    """计算 scaled dot-product attention 的 FLOPs（计算复杂度）。"""
    return json.dumps(_calc_attention_complexity(d_k, seq_len), ensure_ascii=False)
@tool
def calculate_attention_params(d_model:int,n_heads:int)->str:
    """计算多头 attention 模块的参数量。"""
    return json.dumps(_calc_attention_params(d_model, n_heads), ensure_ascii=False)
@tool
def calculate_ffn_params(d_model:int,expansion:int=4)->str:
    """计算 FFN (Feed-Forward Network) 模块的参数量。"""
    return json.dumps(_calc_ffn_params(d_model, expansion), ensure_ascii=False)
@tool
def search_textbook(query:str)->str:
    """在《Build a LLM From Scratch》教材中检索相关概念的原文。当用户问概念、原理、定义时，必须调用此工具。"""
    if _rag_instance is None:
        return "RAG未初始化"
    contexts = _rag_instance.search_multi_query(
        query, top_k=5, n_queries=2, per_query_k=10,
        question_item={"question": query, "category": None, "chapter": None},
        tolerance=1,
    )
    text = "\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in contexts])
    return text if text else "没有找到相关资料。"
def _safe_llm_call(rag:RAG,prompt:str,max_tokens:int=200,temperature:float=0.0)->str:
    try:
        response = rag.llm_client.chat.completions.create(
            model = 'deepseek-chat',
            messages = [{'role':'user','content':prompt}],
            temperature = temperature,
            max_tokens = max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [Graph] LLM 调用失败: {type(e).__name__}: {e}")
        return ""
def _history_to_text(history,last_n:int=4)->str:  #把复杂的 LangChain HumanMessage / AIMessage 对象，提取成纯文本 "user: xxx\nassistant: yyy"，主要用于送去给 LLM 做摘要。
    if not history:
        return ''
    lines = []
    for m in history[-last_n:]:
        if isinstance(m,HumanMessage):
            lines.append(f"user:{m.content}")
        elif isinstance(m,AIMessage):
            lines.append(f"assistant:{m.content}")
        elif isinstance(m,dict):
            lines.append(f"{m.get('role','user')}:{m.get('content','')}")
    return '\n'.join(lines)
class AgentState(TypedDict):
    messages :Annotated[list,operator.add] #这里的 operator.add 意味着“追加”。LangGraph 的底层机制会自动把你新传进去的这条 HumanMessage，追加拼接到数据库里保存的历史 messages 列表的末尾。所以你不需要手动把历史消息再传一遍。
    summary:str #保存长对话的历史摘要
    question:str
    answer:str
    contexts:List[Dict]  # 记录最后一次 search_textbook 拿到的 context，供 UI 渲染
    current_topic : str #记录当前正在讨论的核心主题
    entities:List[str] #预留接口
def build_agent(rag:RAG):
    global _rag_instance
    _rag_instance = rag
    llm_with_tools = ChatOpenAI(
        model = 'deepseek-chat',
        api_key = rag.llm_client.api_key,
        base_url = "https://api.deepseek.com",
        temperature = 0.0
    ).bind_tools([
        calculate_attention_complexity,
        calculate_attention_params,
        calculate_ffn_params,
        search_textbook
    ])
    def agent_node(state:AgentState):
        msgs = state.get('messages',[])
        recent_msgs = msgs[-6:] if len(msgs) > 6 else msgs
        topic_hint = f"【当前讨论主题：{state.get('current_topic', '')}】" if state.get("current_topic") else ""
        summary_hint = f"【历史对话摘要：{state.get('summary', '')}】\n" if state.get("summary") else ""
        system_prompt = f"""你是 CS/NLP 学习助手。{topic_hint}
        1. 如果需要算 FLOPs 或参数量，必须调用提供的工具。
        2. 如果问概念、原理、定义，必须先调用 search_textbook 工具查书。
        3. 基于资料回答，并加上 [Page X] 引用。不要编造。"""
        invoke_msgs = [HumanMessage(content=system_prompt)]+recent_msgs #把系统提示词伪装成“用户说的第一句话”，然后拼接 (+) 上真实的最近历史消息，模型会把它当成一种强烈的上下文指令，执行效果往往更好。
        response = llm_with_tools.invoke(invoke_msgs)
        return {'messages':[response]}
    def tool_node(state:AgentState):
        last_message = state['messages'][-1]
        tool_messages = []
        #初始取历史状态里的contexts
        current_contexts = state.get('contexts',[])
        for tool_call in last_message.tool_calls:
            t_name = tool_call['name']
            args = tool_call['args']
            print(f"  [Tool] 调用了 {t_name}，参数={args}")
            if t_name == 'search_textbook':
                res = search_textbook.invoke(args)# 第一次：调用工具，拿到字符串结果 res
                current_contexts = _rag_instance.search_multi_query(
                    args.get("query", ""), top_k=5, n_queries=2, per_query_k=10,
                    question_item={"question": args.get("query", ""), "category": None, "chapter": None},
                    tolerance=1,) ## 第二次：重新调用底层方法，拿到结构化数据 current_contexts
            #search_textbook.invoke 拿纯文本给 LLM 看；search_multi_query 重新查一次拿结构化字典存进 contexts 给前端看。
            elif t_name == "calculate_attention_complexity":
                res = calculate_attention_complexity.invoke(args)
            elif t_name == "calculate_attention_params":
                res = calculate_attention_params.invoke(args)
            elif t_name == "calculate_ffn_params":
                res = calculate_ffn_params.invoke(args)
            else:
                res = "未知工具"
            tool_messages.append(ToolMessage(content=res, tool_call_id=tool_call["id"]))
        return {'messages': tool_messages, 'contexts': current_contexts}
    def should_continue(state:AgentState):
        last_message = state['messages'][-1]
        if hasattr(last_message,'tool_calls') and last_message.tool_calls:
            return 'tools'
        return 'end'
    def post_process_node(state:AgentState):#后置处理节点，它扮演着“收尾总结”和“记忆提炼”的角色。它的核心任务只有两个：提取最终答案（存起来，供最终输出给用户）。动态更新当前讨论主题（提取核心实体，防止后续对话跑题）。
        last_message = state['messages'][-1]
        answer = last_message.content if hasattr(last_message,'content') else str(last_message)
        topic = state.get('current_topic','') #从全局状态里读取之前记录的主题
        if answer:
            entity_prompt = f"根据以下回答提取一个核心实体（1个词）：\n{answer[:200]}\n实体："
            new_topic = _safe_llm_call(rag, entity_prompt, max_tokens=20)
            if new_topic:
                topic = new_topic
        return {'answer':answer,'current_topic':topic}
    def summarize_node(state:AgentState):
        msgs = state['messages']
        if len(msgs) <= 10:
            return {}
        old = msgs[:-8]
        old_text = _history_to_text(old,last_n=len(old))
        existing = state.get("summary", "")
        prompt = f"把以下对话压缩成不超过 200 字的摘要：\n已有：{existing}\n新增：\n{old_text}\n合并后："
        summary = _safe_llm_call(rag, prompt, max_tokens=300)
        return {"summary": summary} if summary else {}
    #画图
    workflow = StateGraph(AgentState)
    workflow.add_node('agent',agent_node)
    workflow.add_node('tools',tool_node)
    workflow.add_node('post_process',post_process_node)
    workflow.add_node('summarize',summarize_node)
    workflow.set_entry_point('agent')
    workflow.add_conditional_edges('agent',should_continue,{'tools':"tools",'end':'post_process'})
    workflow.add_edge('tools',"agent")
    workflow.add_edge('post_process','summarize')
    workflow.add_edge('summarize',END)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)





