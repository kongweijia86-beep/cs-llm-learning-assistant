"""
Agent Harness Demo (CLI 版本)
该脚本用于在终端中演示 LangGraph Agent 的完整运行过程：
- 意图识别
- Tool Calling (查书 + 计算器)
- 多轮状态流转
"""
from src.rag import RAG
from src.agent_graph import build_agent
from langchain_core.messages import HumanMessage
import uuid
if __name__ == "__main__":
    print("正在初始化 RAG...")
    rag = RAG("data/raw")
    print("正在构建 Agent Harness...")
    agent = build_agent(rag)
    # 分配一个独立的 session_id 记忆
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    question = "如果 d_model 是 768，n_heads 是 12，那么注意力模块的参数量是多少？请查书确认公式。"
    print(f"\n用户提问：{question}")
    # 唤起 Agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    print("\n================ 最终回答 ================")
    print(result["answer"])
    print("\n================ 工具调用记录 ================")
    for msg in result["messages"]:
        if msg.__class__.__name__ == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"- Agent 调用了工具：{tc['name']}，参数：{tc['args']}")

    print("\n================ 检索到的引用 ================")
    for c in result.get("contexts", []):
        print(f"[Page {c['page']}] {c['text'][:100]}...")