import json
import os
import re
import random
import time
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

from openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

from src.rag import RAG
from src.page_filter import _page_to_chapter, _normalize_chapter

# ============ 路径 ============
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PDF_DIR = PROJECT_DIR / "data" / "raw"
QUESTIONS_FILE = BASE_DIR / "questions.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============ 配置 ============
MULTI_QUERY_CATEGORIES = {"cross_chapter", "multihop"}
MIN_EVIDENCE_PAGES = 2
HARD_ABSTAIN_REASONS = {"no_evidence", "too_few_pages"}
CITATION_PATTERN = re.compile(r"\[Page\s+(\d+)\]")
RANDOM_SEED = 42
DEFAULT_TOP_K = 5

# 模式配置：top_k=None 表示跟随 --top_k；写死数字表示该模式固定块数
MODE_CONFIG = {
    "zero_shot": {"retrieve": False, "top_k": 0,    "random": False},
    "random":    {"retrieve": True,  "top_k": None, "random": True},
    "rag_top1":  {"retrieve": True,  "top_k": 1,    "random": False},
    "rag":       {"retrieve": True,  "top_k": None, "random": False},
}
ALL_MODES = ["rag", "rag_top1", "zero_shot", "random"]


# ---------- LLM 重试 ----------
def _call_llm_with_retry(fn, max_retries=3, base_delay=2):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [LLM retry] 第 {attempt+1} 次失败: "
                      f"{type(e).__name__}: {e}，{delay}s 后重试")
                time.sleep(delay)
    raise last_exc


# ---------- 解析 ----------
def resolve_relevant_pages(item):
    """兼容 relevant_pages / page_hints / page_hint 三种写法。"""
    if item.get("relevant_pages"):
        return [{"source": rp["source"], "page": rp["page"]}
                for rp in item["relevant_pages"]]
    source = item.get("source")
    pages = item.get("page_hints") or (
        [item["page_hint"]] if item.get("page_hint") is not None else []
    )
    if not pages:
        raise ValueError(
            f"问题 {item.get('id')} 缺少 relevant_pages / page_hints / page_hint"
        )
    return [{"source": source, "page": p} for p in pages]


def is_hit(retrieved, relevant_pages, tolerance=1):
    for r in retrieved:
        for rel in relevant_pages:
            if r["source"] == rel["source"] and abs(r["page"] - rel["page"]) <= tolerance:
                return True
    return False


def _seed_for_item(item, base_seed=RANDOM_SEED):
    """由题目 id 派生稳定种子：同一题无论跑多少次/什么顺序，抽样一致。"""
    key = f"{base_seed}:{item.get('id', '')}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ---------- LLM 客户端 ----------
def get_llm_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ---------- 拒答判断 ----------
def check_evidence_sufficiency(merged, question_item):
    if not merged:
        return False, "no_evidence"

    chapter = question_item.get("chapter", "") or ""
    missing_chapters = []
    if "&" in chapter:
        required = {_normalize_chapter(ch) for ch in chapter.split("&") if _normalize_chapter(ch)}
        covered = {_page_to_chapter(r["page"]) for r in merged if _page_to_chapter(r["page"])}
        missing = required - covered
        if missing:
            missing_chapters = sorted(missing)

    if len(merged) < MIN_EVIDENCE_PAGES:
        return False, "too_few_pages"

    if missing_chapters:
        return True, f"ok_with_missing_chapters: {missing_chapters}"
    return True, "ok"


# ---------- 生成 ----------
def generate_answer(question, retrieved_chunks, llm_client,
                    model="deepseek-chat", history=None):
    if not retrieved_chunks:
        context = "(no context available)"
    else:
        context = "\n\n".join(
            f"[Page {c['page']}]\n{c['text']}" for c in retrieved_chunks
        )

    prompt = f"""你是一个 CS/NLP 学习助手。请根据下面的资料回答问题。

资料：
{context}

问题：{question}

要求：
1. 优先依据资料回答。
2. 每个主要结论后用 [Page X] 标注来源，如 [Page 23]。
3. 如果资料中没有足够的信息来回答问题，必须明确说：
   "资料中没有找到足够的信息来回答这个问题。"
   不要尝试用你自己的知识补充。
4. 如果资料之间有冲突，分别说明，不要强行合并。
5. 只允许引用上面出现的 [Page X]。不要虚构引用编号。

回答格式：
- 如果资料足够：先给答案，再给引用。
- 如果资料不足：直接说资料不足，不要编。

Answer:"""

    messages = []
    for msg in (history or []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    messages.append({"role": "user", "content": prompt})

    def _call():
        return llm_client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=500,
        )

    resp = _call_llm_with_retry(_call, max_retries=3)
    return resp.choices[0].message.content.strip()


# ---------- 引用校验 ----------
def validate_citations(generated_answer, retrieved_chunks):
    # findall 在单捕获组时返回字符串列表，int() 转换即可
    cited_pages = [int(m) for m in CITATION_PATTERN.findall(generated_answer)]
    if not cited_pages:
        return True, "no_citation", [], []

    available = {c["page"] for c in retrieved_chunks}
    missing = [p for p in cited_pages if p not in available]
    if missing:
        return (False,
                f"cited_pages_not_in_context: {sorted(set(missing))}",
                cited_pages, missing)
    return True, "ok", cited_pages, []


# ---------- 评判 ----------
def judge_answer(question, reference_answer, generated_answer,
                 retrieved_chunks, llm_client, model="deepseek-chat"):
    context_summary = "\n".join(
        f"[Page {c['page']}]" for c in retrieved_chunks
    ) if retrieved_chunks else "(no context)"

    prompt = f"""你是一个严格的评审。请评估下面这个 RAG 回答的质量。

问题：{question}

参考答案：{reference_answer}

系统检索到的页面：
{context_summary}

系统回答：{generated_answer}

请判断这个回答属于哪一类：
- correct：回答正确，覆盖了参考答案的关键点
- partial：部分正确，但漏了关键信息
- wrong：回答错误，或者与参考答案矛盾
- abstained：系统明确说"资料中没有找到相关内容"，且检索到的页面确实不足以回答问题
- error：系统报错或回答为空

注意：
- 如果系统说"资料不足"，但检索到的页面其实包含了参考答案的关键信息，应该判 wrong（这是错误拒答），不是 abstained。
- 如果系统说"资料不足"，且检索到的页面确实不包含关键信息，判 abstained。

只输出 JSON：
{{"label": "correct/partial/wrong/abstained/error", "reason": "简短说明"}}
"""

    def _call():
        return llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

    try:
        resp = _call_llm_with_retry(_call, max_retries=3)
        result = json.loads(resp.choices[0].message.content)
        return result.get("label", "wrong"), result.get("reason", "")
    except Exception as e:
        print(f"  [judge_answer] 评判失败: {e}")
        return "judge_error", f"judge error: {e}"


# ---------- 检索入口 ----------
def _ensure_text(results, rag):
    """给缺 text 的结果补同页文本；索引缓存到 rag，避免每题重建。"""
    index = getattr(rag, "_page_text_index", None)
    if index is None:
        index = {}
        for c in rag.chunks:
            index.setdefault((c["source"], c["page"]), c["text"])
        rag._page_text_index = index
    for r in results:
        if not r.get("text"):
            r["text"] = index.get(
                (r["source"], r["page"]),
                f"(page {r['page']} content not available)",
            )


def retrieve(rag, question_item, top_k=DEFAULT_TOP_K, use_random=False,
             seed=RANDOM_SEED):
    q = question_item["question"]
    category = question_item.get("category")

    if use_random:
        # 每题独立 Random 实例，种子由题目 id 派生：可复现且题目间不同
        rng = random.Random(_seed_for_item(question_item, seed))
        sampled = rng.sample(rag.chunks, min(top_k, len(rag.chunks)))
        return [
            {"text": c["text"], "page": c["page"],
             "source": c["source"], "score": 0.0}
            for c in sampled
        ]

    if category in MULTI_QUERY_CATEGORIES:
        results = rag.search_multi_query(
            q, top_k=top_k, n_queries=3, per_query_k=10,
            question_item=question_item, tolerance=1,
        )
    else:
        query_emb = rag.embedding_model.encode([q])[0]
        results = rag.search(
            query_emb, query_text=q, top_k=top_k,
            question_item=question_item, tolerance=1,
        )

    _ensure_text(results, rag)
    return results


# ---------- 主评估 ----------
def evaluate_generation(questions_file, rag, llm_client,
                        mode="rag", top_k=DEFAULT_TOP_K, tolerance=1,
                        num_questions=None, seed=RANDOM_SEED):
    if mode not in MODE_CONFIG:
        raise ValueError(f"未知 mode: {mode}，可选: {list(MODE_CONFIG)}")

    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if num_questions is not None:
        data = data[:num_questions]

    cfg = MODE_CONFIG[mode]
    results = []

    for i, item in enumerate(data):
        q = item["question"]
        relevant_pages = resolve_relevant_pages(item)
        category = item.get("category")

        # 1. 检索
        if not cfg["retrieve"]:
            retrieved, hit = [], False
        else:
            k = cfg["top_k"] if cfg["top_k"] is not None else top_k
            retrieved = retrieve(rag, item, top_k=k,
                                 use_random=cfg["random"], seed=seed)
            hit = is_hit(retrieved, relevant_pages, tolerance)

        # 2. 证据充足性
        is_sufficient, suff_reason = check_evidence_sufficiency(retrieved, item)

        if suff_reason in HARD_ABSTAIN_REASONS:
            generated = f"资料中没有找到足够的信息来回答这个问题。({suff_reason})"
            verdict, reason = "abstained", f"abstained: {suff_reason}"
            citation_valid, cited_pages = None, []
        else:
            try:
                generated = generate_answer(q, retrieved, llm_client)
                citation_valid, cit_reason, cited_pages, _ = validate_citations(
                    generated, retrieved
                )
                if not citation_valid:
                    verdict, reason = "abstained", f"invalid_citation: {cit_reason}"
                else:
                    verdict, reason = judge_answer(
                        q, item["reference_answer"], generated,
                        retrieved, llm_client,
                    )
            except Exception as e:
                print(f"  [SKIP] {item['id']} 生成失败: {type(e).__name__}: {e}")
                generated = ""
                verdict, reason = "error", f"generation error: {type(e).__name__}: {e}"
                citation_valid, cited_pages = None, []

        results.append({
            "id": item["id"],
            "category": category,
            "chapter": item.get("chapter"),
            "question": q,
            "reference_answer": item["reference_answer"],
            "retrieval_hit": hit,
            "retrieved_pages": [r["page"] for r in retrieved],
            "evidence_sufficient": is_sufficient,
            "evidence_reason": suff_reason,
            "citation_valid": citation_valid,
            "cited_pages": cited_pages,
            "generated": generated,
            "verdict": verdict,
            "reason": reason,
        })

        mark = "✅" if hit else "❌"
        print(f"[{i+1}/{len(data)}] {item['id']:24s} 检索{mark} 生成[{verdict}]")

    return results


# ---------- 汇总 ----------
def summarize(results, mode_name=""):
    total = len(results)
    vc = Counter(r["verdict"] for r in results)
    correct, partial, wrong = vc["correct"], vc["partial"], vc["wrong"]
    abstained, error, judge_error = vc["abstained"], vc["error"], vc["judge_error"]

    cc = Counter(r.get("citation_valid") for r in results)
    valid_citation, invalid_citation, no_citation = cc[True], cc[False], cc["no_citation"]

    print("\n===== 引用校验统计 =====")
    print(f"  有引用且合法: {valid_citation}")
    print(f"  无引用: {no_citation}")
    print(f"  引用了不存在的页: {invalid_citation}")
    print(f"\n{'='*60}")
    print(f"===== 生成质量汇总 [{mode_name}] =====")
    print(f"{'='*60}")
    print(f"总数:        {total}")
    print(f"correct:     {correct}/{total} = {correct/total:.2%}")
    print(f"partial:     {partial}/{total} = {partial/total:.2%}")
    print(f"wrong:       {wrong}/{total} = {wrong/total:.2%}")
    print(f"abstained:   {abstained}/{total} = {abstained/total:.2%}")
    print(f"error:       {error}/{total} = {error/total:.2%}")
    print(f"judge_error: {judge_error}/{total} = {judge_error/total:.2%}")

    valid = total - error - judge_error
    if valid > 0:
        print(f"有效总数:     {valid}")
        print(f"（correct + partial）/ valid = {(correct+partial)/valid:.2%}")

    if abstained:
        print("\n===== 拒答原因分布 =====")
        for reason, cnt in Counter(
            r["reason"] for r in results if r["verdict"] == "abstained"
        ).most_common():
            print(f"  {reason}: {cnt}")
        print("\n===== 拒答明细 =====")
        for r in results:
            if r["verdict"] == "abstained":
                print(f"  {r['id']}: {r['reason'][:100]}")

    missing_items = [
        r for r in results if "missing_chapters" in (r.get("evidence_reason") or "")
    ]
    if missing_items:
        print(f"\n===== 缺章节但继续生成的题 ({len(missing_items)} 条) =====")
        for r in missing_items:
            print(f"  {r['id']}: {r['evidence_reason']} -> {r['verdict']}")

    print("\n===== 检索命中 × 生成质量 =====")
    ct = defaultdict(Counter)
    for r in results:
        ct[r["retrieval_hit"]][r["verdict"]] += 1
    for hit in (True, False):
        row = ct[hit]
        n = sum(row.values())
        if not n:
            continue
        print(f"\n检索{'命中' if hit else '未命中'} ({n} 条):")
        for v in ("correct", "partial", "wrong", "abstained", "error", "judge_error"):
            if row[v]:
                print(f"  {v}: {row[v]} ({row[v]/n:.1%})")

    print("\n===== 按 category 分组 =====")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    print(f"{'category':16s}  {'n':>4s}  {'correct':>10s}  {'partial':>10s}  "
          f"{'wrong':>10s}  {'abstained':>10s}  {'error':>8s}")
    for cat, items in sorted(by_cat.items()):
        n = len(items)
        c = Counter(x["verdict"] for x in items)
        e = c["error"] + c["judge_error"]
        print(f"{cat:16s}  {n:>4d}  {c['correct']/n:>10.1%}  {c['partial']/n:>10.1%}  "
              f"{c['wrong']/n:>10.1%}  {c['abstained']/n:>10.1%}  {e:>8d}")

    return {
        "mode": mode_name, "total": total,
        "correct": correct/total, "partial": partial/total,
        "wrong": wrong/total, "abstained": abstained/total,
        "error": error/total, "judge_error": judge_error/total,
    }


def compare_modes(all_summaries):
    print(f"\n{'='*100}")
    print("===== 跨模式对比 =====")
    print(f"{'='*100}")
    print(f"{'mode':14s}  {'correct':>10s}  {'partial':>10s}  {'wrong':>10s}  "
          f"{'abstained':>10s}  {'error':>8s}  {'judge_err':>10s}  {'c+p':>10s}")
    for s in all_summaries:
        print(
            f"{s['mode']:14s}  "
            f"{s['correct']:>10.1%}  {s['partial']:>10.1%}  "
            f"{s['wrong']:>10.1%}  {s['abstained']:>10.1%}  "
            f"{s['error']:>8.1%}  {s['judge_error']:>10.1%}  "
            f"{s['correct']+s['partial']:>10.1%}"
        )


# ---------- 入口 ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="端到端生成评估")
    parser.add_argument("--mode", type=str, default="rag",
                        choices=ALL_MODES + ["all"],
                        help="评估模式，all 表示跑全部 4 组")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help="rag/random 模式取几块（rag_top1 固定 1，不受影响）")
    parser.add_argument("--tolerance", type=int, default=1)
    parser.add_argument("--num_questions", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="随机种子，固定后 random 模式可复现")
    args = parser.parse_args()

    print(f"BASE_DIR       = {BASE_DIR}")
    print(f"PDF_DIR        = {PDF_DIR} exists={PDF_DIR.exists()}")
    print(f"QUESTIONS_FILE = {QUESTIONS_FILE} exists={QUESTIONS_FILE.exists()}")
    print(f"mode           = {args.mode}")
    print(f"top_k          = {args.top_k}")
    print(f"tolerance      = {args.tolerance}")
    print(f"seed           = {args.seed}")

    rag = RAG(str(PDF_DIR))
    llm_client = get_llm_client()

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    all_summaries = []

    for mode in modes:
        print(f"\n\n{'#'*60}")
        print(f"# 开始评估 mode = {mode}")
        print(f"{'#'*60}\n")

        results = evaluate_generation(
            str(QUESTIONS_FILE), rag, llm_client,
            mode=mode, top_k=args.top_k,
            tolerance=args.tolerance,
            num_questions=args.num_questions,
            seed=args.seed,
        )
        summary = summarize(results, mode_name=mode)
        all_summaries.append(summary)

        out_file = Path(args.output) if args.output else RESULTS_DIR / f"generation_{mode}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "mode": mode,
                    "top_k": args.top_k,
                    "tolerance": args.tolerance,
                    "seed": args.seed,
                    "num_questions": args.num_questions,
                    "model": "deepseek-chat",
                    "temperature_generate": 0.2,
                    "temperature_judge": 0.0,
                },
                "summary": summary,
                "details": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {out_file}")

    if len(all_summaries) > 1:
        compare_modes(all_summaries)