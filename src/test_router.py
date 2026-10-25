# #我的 Router 单测四个分支全过。特别值得注意的是 calculator 那个 case——问题里同时出现了 ‘attention’（检索关键词）和 ‘d_k=512’（计算关键词），Router 仍然正确选了 calculator，说明 LLM 路由不是简单关键词匹配，而是理解了‘这个问题要算数，不是要查定义’。同时 retrieve 和 compare 分支的 multi-query 子问题拆解也符合预期：compare 把‘对比 LayerNorm 和 BatchNorm’拆成‘各自定义 + 差异’，这是对比类问题的标准拆法。”
# import sys
# from pathlib import Path
#
# # ★ 1. 把项目根加进 sys.path，否则 `from src.rag import RAG` 会 ImportError
# BASE_DIR = Path(__file__).resolve().parent.parent  # src/test_router.py -> 项目根
# sys.path.insert(0, str(BASE_DIR))
#
# # ★ 2. 必须在 sys.path 之后 import
# from src.rag import RAG
# from src.agent_graph import build_agent
#
# PDF_DIR = BASE_DIR / "data" / "raw"
#
#
# def test_router_four_branches():
#     # ★ 3. RAG 初始化放函数内，避免 pytest 收集阶段触发重初始化
#     rag = RAG(str(PDF_DIR))
#     agent = build_agent(rag)
#
#     cases = [
#         ("What is self-attention?", "retrieve"),
#         ("如果 d_k=512、n=1024，attention 的 FLOPs 是多少？", "calculator"),
#         ("Compare LayerNorm and BatchNorm", "compare"),
#         ("你好", "direct_answer"),
#     ]
#
#     for q, expected in cases:
#         config = {"configurable": {"thread_id": f"test_{expected}"}}
#         out = agent.invoke(
#             {
#                 "question": q,
#                 "messages": [],
#                 "retry_count": 0,
#                 "max_retries": 2,
#                 "route": "",
#                 "tool_result": None,
#             },
#             config=config,
#         )
#         actual = out.get("route")
#         print(f"Q: {q[:40]:40s} | route={actual:15s} | expected={expected:15s} | {'✅' if actual == expected else '❌'}")
#         assert actual == expected, f"路由错误: '{q}' -> {actual}, 期望 {expected}"
# src/test_router.py

import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # src/test_router.py -> 项目根
sys.path.insert(0, str(BASE_DIR))

from src.rag import RAG
from src.agent_graph import build_agent

PDF_DIR = BASE_DIR / "data" / "raw"

CITATION_RE = re.compile(r"\[Page\s+(\d+)\]")


def _run_agent(agent, q, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    return agent.invoke(
        {
            "question": q,
            "messages": [],
            "retry_count": 0,
            "max_retries": 2,
            "route": "",
            "tool_result": None,
        },
        config=config,
    )


def _print_out(q, expected, out):
    actual = out.get("route")
    answer = out.get("answer", "") or ""
    contexts = out.get("contexts", []) or []
    pages = sorted(set(c["page"] for c in contexts)) if contexts else []
    mark = "✅" if actual == expected else "❌"
    print(f"\n{'='*80}")
    print(f"Q: {q}")
    print(f"  route={actual} | expected={expected} | {mark}")
    print(f"  contexts={len(contexts)} 页 | pages={pages[:10]}{'...' if len(pages)>10 else ''}")
    print(f"  answer[:120]={answer[:120]!r}")
    return actual, answer, contexts, pages


# 顶部加：
from langchain_core.messages import HumanMessage


def test_multi_turn_coreference():
    rag = RAG(str(PDF_DIR))
    agent = build_agent(rag)
    config = {"configurable": {"thread_id": "test_multi_turn"}}

    # ---------- 第一轮：建立指代对象 ----------
    q1 = "What is self-attention?"
    out1 = agent.invoke(
        {
            "question": q1,
            "messages": [HumanMessage(content=q1)],
            "retry_count": 0,
            "max_retries": 2,
            "route": "",
            "tool_result": None,
        },
        config=config,
    )
    route1 = out1.get("route")
    print(f"\n第一轮 Q: {q1} | route={route1}")
    assert route1 == "retrieve", f"第一轮路由错误: {route1}"

    # ---------- 第二轮：用“它”指代 ----------
    q2 = "它的计算复杂度是多少？"
    out2 = agent.invoke(
        {
            "question": q2,
            "messages": [HumanMessage(content=q2)],
            "retry_count": 0,
            "max_retries": 2,
            "route": "",
            "tool_result": None,
        },
        config=config,
    )
    route2 = out2.get("route")
    answer2 = out2.get("answer", "") or ""
    contexts2 = out2.get("contexts", []) or []
    print(f"第二轮 Q: {q2} | route={route2}")
    print(f"  contexts={len(contexts2)} 页 | pages={sorted(set(c['page'] for c in contexts2))[:10]}")
    print(f"  answer[:150]={answer2[:150]!r}")

    # ---------- 校验：第二轮要么走 calculator（理解“它”= self-attention 的复杂度），
    #            要么走 retrieve 但检索到 self-attention 的页 ----------
    assert route2 in ("calculator", "retrieve"), f"第二轮路由异常: {route2}"

    if route2 == "retrieve":
        # 检索到的页里应该有 Ch3 的 self-attention 页（69/70 附近）
        pages2 = set(c["page"] for c in contexts2)
        ch3_pages = {p for p in pages2 if 65 <= p <= 115}  # Ch3 大致范围
        assert ch3_pages, f"第二轮 retrieve 没检索到 Ch3 self-attention 页，pages={sorted(pages2)}"
        print(f"  ✅ 第二轮 retrieve 命中 Ch3 页: {sorted(ch3_pages)}")
    else:
        # calculator 分支：answer 里应有复杂度数字
        assert any(k in answer2 for k in ["n^2", "O(n", "复杂度", "FLOPs", "平方"]), \
            f"calculator 答案里没看到复杂度相关内容: {answer2[:200]}"
        print(f"  ✅ 第二轮 calculator 返回复杂度相关答案")

    print(f"\n{'='*80}")
    print("🎉 Step 3 通过：多轮指代消解生效")
def test_multi_turn_coreference():
    rag = RAG(str(PDF_DIR))
    agent = build_agent(rag)

    # ========== 场景 A：第二轮走 retrieve，验证指代消解 ==========
    config_a = {"configurable": {"thread_id": "test_coref_retrieve"}}

    q1 = "What is self-attention?"
    out1 = agent.invoke({
        "question": q1, "messages": [HumanMessage(content=q1)],
        "retry_count": 0, "max_retries": 2, "route": "", "tool_result": None,
    }, config=config_a)
    print(f"\n[A-1] Q: {q1} | route={out1.get('route')}")
    assert out1.get("route") == "retrieve"

    # ★ 第二轮问一个“必须查书”的指代问题
    q2 = "它在 Transformer 里的具体作用是什么？"
    out2 = agent.invoke({
        "question": q2, "messages": [HumanMessage(content=q2)],
        "retry_count": 0, "max_retries": 2, "route": "", "tool_result": None,
    }, config=config_a)
    route2 = out2.get("route")
    contexts2 = out2.get("contexts", []) or []
    pages2 = sorted(set(c["page"] for c in contexts2))
    print(f"[A-2] Q: {q2} | route={route2} | pages={pages2[:10]}")
    assert route2 in ("retrieve", "compare"), f"第二轮应走检索类分支，实际 {route2}"
    # 检索到的页里必须有 Ch3 的 self-attention 页
    ch3_pages = {p for p in pages2 if 65 <= p <= 115}
    assert ch3_pages, f"第二轮没检索到 Ch3 self-attention 页，pages={pages2}"
    print(f"  ✅ 场景 A 通过：第二轮检索命中 Ch3 页 {sorted(ch3_pages)}")

    # ========== 场景 B：第二轮走 calculator，验证上下文指代 ==========
    config_b = {"configurable": {"thread_id": "test_coref_calc"}}

    q3 = "What is self-attention?"
    agent.invoke({
        "question": q3, "messages": [HumanMessage(content=q3)],
        "retry_count": 0, "max_retries": 2, "route": "", "tool_result": None,
    }, config=config_b)

    q4 = "它的计算复杂度是多少？"
    out4 = agent.invoke({
        "question": q4, "messages": [HumanMessage(content=q4)],
        "retry_count": 0, "max_retries": 2, "route": "", "tool_result": None,
    }, config=config_b)
    route4 = out4.get("route")
    answer4 = out4.get("answer", "") or ""
    print(f"[B-2] Q: {q4} | route={route4}")
    assert route4 == "calculator"
    assert any(k in answer4 for k in ["n^2", "O(n", "FLOPs", "复杂度"]), \
        (f"calculator 答案无复杂度内容: {answer4[:200]}")
    print(f"  ✅ 场景 B 通过：calculator 返回复杂度答案")