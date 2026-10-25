import json
import numpy as np
from pathlib import Path
from src.rag import RAG
from collections import defaultdict
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PDF_DIR = PROJECT_DIR/'data'/'raw'
QUESTIONS_FILE = BASE_DIR/'questions.json'
RESULTS_DIR = BASE_DIR/'results'
RESULTS_FILE = RESULTS_DIR/'baseline.json'
MULTI_QUERY_CATEGORIES = {'cross chapter','definition','multihop','derivation','code'}
def resolve_relevant_pages(item):
    if 'relevant pages' in item and item['relevant pages']:
        return[
            {'source':rp['source'],'page':rp['page'] } for rp in item['relevant pages']
        ]
    if 'page_hints' in item and item['page_hints']:
        source = item.get('source')
        if source is None:
            raise ValueError(f"问题{item.get('id')} 使用page_hints 但缺少sources")
        return [{'source':source,'page':p} for p in item['page_hints']]
    if 'page_hint' in item and item['page_hint']:
        source = item.get('source')
        if source is None:
            raise ValueError(f"问题{item.get('id')} 使用page_hint但缺少source")
        return [{'source':source,'page':item['page_hint']}]
    raise ValueError(f"问题 {item.get('id')} 缺少 relevant_pages / page_hints / page_hint")
def _matched_pages(retrieved,relevant_pages,tolerance):
    matched = set()
    for item in retrieved:
        for rel in relevant_pages:
            if item['source'] == rel['source'] and abs(item['page']-rel['page']) <= tolerance:
                matched.add((rel['source'],rel['page']))
    return matched
def is_hit(retrieved,relevant_pages,hit_mode,tolerance=0):
    matched = _matched_pages(retrieved,relevant_pages,tolerance)
    total = len({(r["source"], r["page"]) for r in relevant_pages})
    if hit_mode == 'any':
        return len(matched) >= 1
    elif hit_mode == 'all':
        return len(matched) == total
    elif hit_mode == 'at_least_2':
        return len(matched) >= 2
    return False
def hit_at_k(retrieved,relevant_pages,hit_mode,k,tolerance=1):
    """只看retrieved的前k个，按hit_mode判断"""
    return 1.0 if is_hit(retrieved[:k],relevant_pages,hit_mode,tolerance) else 0.0
def mrr(retrieved,relevant_pages,hit_mode,tolerance=0):
    matched = set()
    total = len({(r['source'],r['page']) for r in relevant_pages})
    needed = total if hit_mode == 'all' else(2 if hit_mode == 'at_least_2' else 1)
    for i,item in enumerate(retrieved):
        for rel in relevant_pages:
            if item['source'] == rel['source'] and abs(item['page']-rel['page'])<=tolerance:
                matched.add((rel['source'],rel['page']))
        if len(matched) >= needed:
            return 1/(i+1)
    return 0.0
def evaluate(questions_file,rag,top_k=5,tolerance=0,use_multi_query=True):
    with open(questions_file,'r',encoding='utf-8') as f:
        data = json.load(f)
    hits_1_any,hits_3_any,hits_5_any = [],[],[]
    hits_1_all,hits_3_all,hits_5_all = [],[],[]
    hits_3_multi,hits_5_multi = [],[]
    mrrs_any,mrrs_all,mrrs_multi = [],[],[]
    details = []
    num_multi_hop = 0
    for item in data:
        if 'source' not in item:
            item['source'] =  "Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf"
        q = item['question']
        relevant_pages = resolve_relevant_pages(item)
        category = item.get('category')
        if use_multi_query and category in MULTI_QUERY_CATEGORIES:
            print(f"  [MQ] {item['id']} (category={category})")
            retrieved = rag.search_multi_query(
                q,top_k=top_k,n_queries=3,per_query_k=10,question_item=item,tolerance=tolerance
            )
        else:
            query_emb = rag.embedding_model.encode([q])[0] #【0】指的是取第0条文本的向量
            retrieved = rag.search(query_emb,query_text=q,top_k=top_k,question_item=item,tolerance=tolerance)
        # any
        h1_any = hit_at_k(retrieved, relevant_pages, "any", k=1, tolerance=tolerance)
        h3_any = hit_at_k(retrieved, relevant_pages, "any", k=3, tolerance=tolerance)
        h5_any = hit_at_k(retrieved, relevant_pages, "any", k=5, tolerance=tolerance)
        m_any = mrr(retrieved, relevant_pages, "any", tolerance=tolerance)
        # all
        h1_all = hit_at_k(retrieved, relevant_pages, "all", k=1, tolerance=tolerance)
        h3_all = hit_at_k(retrieved, relevant_pages, "all", k=3, tolerance=tolerance)
        h5_all = hit_at_k(retrieved, relevant_pages, "all", k=5, tolerance=tolerance)
        m_all = mrr(retrieved, relevant_pages, "all", tolerance=tolerance)
        #at_least_2(只对多跳题有意义)
        is_multi_hop = len({(r['source'],r['page']) for r in relevant_pages}) >= 2
        if is_multi_hop:
            num_multi_hop += 1
            h3_multi = hit_at_k(retrieved,relevant_pages,'at_least_2',k=3,tolerance=tolerance)
            h5_multi = hit_at_k(retrieved,relevant_pages,'at_least_2',k=5,tolerance=tolerance)
            m_multi = mrr(retrieved,relevant_pages,'at_least_2',tolerance=tolerance)
            hits_3_multi.append(h3_multi)
            hits_5_multi.append(h5_multi)
            mrrs_multi.append(m_multi)
        else:
            h3_multi = h5_multi = m_multi = None
        hits_1_any.append(h1_any)
        hits_3_any.append(h3_any)
        hits_5_any.append(h5_any)
        hits_1_all.append(h1_all)
        hits_3_all.append(h3_all)
        hits_5_all.append(h5_all)
        mrrs_any.append(m_any)
        mrrs_all.append(m_all)
        details.append({
            "id": item["id"],
            "chapter": item.get("chapter"),
            "category": category,
            "question": q,
            "relevant_pages": relevant_pages,
            "hit@1_any": h1_any,
            "hit@3_any": h3_any,
            "hit@5_any": h5_any,
            "mrr_any": m_any,
            "hit@1_all": h1_all,
            "hit@3_all": h3_all,
            "hit@5_all": h5_all,
            "mrr_all": m_all,
            "is_multi_hop": is_multi_hop,
            "hit@3_multi": h3_multi,
            "hit@5_multi": h5_multi,
            "mrr_multi": m_multi,
            "retrieved": [
                {"source": r["source"], "page": r["page"], "score": round(r["score"], 4)}
                for r in retrieved]
        })
        # 控制台打印
        status_any = "✅" if h3_any == 1.0 else "❌"
        status_all = "✅" if h3_all == 1.0 else "❌"
        if is_multi_hop:
            status_multi = "✅" if h3_multi == 1.0 else "❌"
            tag = f" {status_multi}[multi]"
        else:
            tag = " [--]"
        print(f"{status_any}[any] {status_all}[all]{tag} [{item['id']}] {q[:50]}")
        if h3_any == 0.0 and h3_all == 0.0:
            expected_str = [(rp["source"], rp["page"]) for rp in relevant_pages]
            print(f"      期望页: {expected_str}")
            print(f"      检索: {[(r['source'], r['page'], round(r['score'], 3)) for r in retrieved[:3]]}")
    summary = {
        "num_questions": len(data),
        "Hit@1_any": float(np.mean(hits_1_any)),
        "Hit@3_any": float(np.mean(hits_3_any)),
        "Hit@5_any": float(np.mean(hits_5_any)),
        "MRR_any": float(np.mean(mrrs_any)),
        "Hit@1_all": float(np.mean(hits_1_all)),
        "Hit@3_all": float(np.mean(hits_3_all)),
        "Hit@5_all": float(np.mean(hits_5_all)),
        "MRR_all": float(np.mean(mrrs_all)),
        "num_multi_hop": num_multi_hop,
        "Hit@3_multi": float(np.mean(hits_3_multi)) if hits_3_multi else 0.0,
        "Hit@5_multi": float(np.mean(hits_5_multi)) if hits_5_multi else 0.0,
        "MRR_multi": float(np.mean(mrrs_multi)) if mrrs_multi else 0.0,
        "tolerance": tolerance,
        "top_k": top_k,
        "use_multi_query": use_multi_query,
    }
    print("\n========== 汇总 ==========")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:12s}: {v:.4f}")
        else:
            print(f"{k:12s}: {v}")
    print("\n========== 按 category 分组 ==========")
    by_cat = defaultdict(list)
    for d in details:
        by_cat[d['category']].append(
            (d['hit@3_any'], d['hit@3_all'], d.get('hit@3_multi'), d.get('is_multi_hop'))
        )
    print(f"{'category':14s}  {'Hit@3_any':>10s}  {'Hit@3_all':>10s}  {'Hit@3_multi':>12s}  {'n_multi':>8s}  {'n':>4s}")
    for cat, vals in sorted(by_cat.items()):
        any_mean = np.mean([v[0] for v in vals])
        all_mean = np.mean([v[1] for v in vals])
        multi_vals = [v[2] for v in vals if v[3] and v[2] is not None]
        multi_mean = np.mean(multi_vals) if multi_vals else float("nan")
        n_multi = len(multi_vals)
        print(f"{cat:14s}  {any_mean:10.4f}  {all_mean:10.4f}  {multi_mean:12.4f}  {n_multi:>8d}  {len(vals):>4d}")
    print("\n========== 按 category 分组（只看 any 失败）==========")
    fail_any = defaultdict(list)
    for d in details:
        if d["hit@3_any"] == 0.0:
            fail_any[d["category"]].append(d["id"])
    for cat, ids in sorted(fail_any.items()):
        print(f"{cat:14s}: {len(ids)} 条失败 -> {ids}")
    print("\n========== 按 category 分组（只看 multi 失败）==========")
    fail_multi = defaultdict(list)
    for d in details:
        if d.get("is_multi_hop") and d.get("hit@3_multi") == 0.0:
            fail_multi[d["category"]].append(d["id"])
    for cat, ids in sorted(fail_multi.items()):
        print(f"{cat:14s}: {len(ids)} 条失败 -> {ids}")
    return summary, details
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="RAG 检索评估（方案 A：只对复杂题走 Multi-Query）")
    parser.add_argument('--no_multi_query',action='store_true',help='禁用Multi-Query(全部走单查询，作为对照组)')
    parser.add_argument('--tolerance',type=int,default=1)
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--output',type=str,default=None,help="结果输出路径，默认 results/baseline_{mq|single}.json")
    args = parser.parse_args()
    if args.output:
        out_file = Path(args.output)
    else:
        suffix = 'single' if args.no_multi_query else 'mq'
        out_file = RESULTS_DIR/f"baseline_{suffix}.json"
    print("BASE_DIR         =", BASE_DIR)
    print("PDF_DIR          =", PDF_DIR, "exists =", PDF_DIR.exists())
    print("QUESTIONS_FILE   =", QUESTIONS_FILE, "exists =", QUESTIONS_FILE.exists())
    print("RESULTS_FILE     =", out_file)
    print("use_multi_query  =", not args.no_multi_query)
    print("tolerance        =", args.tolerance)
    print("top_k            =", args.top_k)
    print("MQ categories    =", MULTI_QUERY_CATEGORIES)
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF 目录不存在: {PDF_DIR}")
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"问题文件不存在: {QUESTIONS_FILE}")
    rag = RAG(str(PDF_DIR))
    summary,details = evaluate(str(QUESTIONS_FILE),rag,top_k=args.top_k,tolerance=args.tolerance,use_multi_query=(not args.no_multi_query))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {out_file}")











        