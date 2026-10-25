import json
from pathlib import Path
import numpy as np
from src.rag import RAG
BASE_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = BASE_DIR / "eval" / "questions.json"
def dcg(relevances, k):
    relevances = np.asarray(relevances)[:k]
    if relevances.size == 0: return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)
def ndcg_at_k(relevances, k):
    ideal_relevances = sorted(relevances, reverse=True)
    dcg_val = dcg(relevances, k)
    idcg_val = dcg(ideal_relevances, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0
def resolve_relevant_pages(item):
    if "relevant_pages" in item and item["relevant_pages"]:
        return [{"source": rp["source"], "page": rp["page"]} for rp in item["relevant_pages"]]
    if "page_hints" in item and item["page_hints"]:
        source = item.get("source", "Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf")
        return [{"source": source, "page": p} for p in item["page_hints"]]
    if "page_hint" in item and item["page_hint"] is not None:
        source = item.get("source", "Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf")
        return [{"source": source, "page": item["page_hint"]}]
    return []
def run_ablation():
    rag = RAG(str(BASE_DIR / "data" / "raw"))
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = {
        "recall_pre": [], "recall_post": [],
        "mrr_pre": [], "mrr_post": [],
        "ndcg_pre": [], "ndcg_post": []
    }
    for item in data:
        q = item["question"]
        rels = resolve_relevant_pages(item)
        for rp in rels:
            if "source" not in rp:
                rp["source"] = "Build_a_Large_Language_Model_(From_Scratch)_v8_MEAP.pdf"
        relevant_pages = set((rp["source"], rp["page"]) for rp in rels)
        query_emb = rag.embedding_model.encode([q])[0]
        #  纯向量召回 Top-20
        candidates_pre = rag.retriever.search(query_emb, top_k=20)
        #  Reranker 精排 Top-5
        candidates_post = rag.reranker.rerank(q, candidates_pre, top_k=5)

        # Recall@20
        retrieved_pre_set = set((r["source"], r["page"]) for r in candidates_pre)
        recall_pre = len(retrieved_pre_set & relevant_pages) / len(relevant_pages) if relevant_pages else 0.0
        metrics["recall_pre"].append(recall_pre)

        # --- MRR/NDCG 重排前 ---
        mrr_pre = 0.0
        relevances_pre = []
        for i, r in enumerate(candidates_pre):
            is_rel = (r["source"], r["page"]) in relevant_pages
            relevances_pre.append(1 if is_rel else 0)
            if is_rel and mrr_pre == 0.0: mrr_pre = 1.0 / (i + 1)
        metrics["mrr_pre"].append(mrr_pre)
        metrics["ndcg_pre"].append(ndcg_at_k(relevances_pre, k=10))

        # --- MRR/NDCG 重排后 ---
        mrr_post = 0.0
        relevances_post = []
        for i, r in enumerate(candidates_post):
            is_rel = (r["source"], r["page"]) in relevant_pages
            relevances_post.append(1 if is_rel else 0)
            if is_rel and mrr_post == 0.0: mrr_post = 1.0 / (i + 1)
        metrics["mrr_post"].append(mrr_post)
        metrics["ndcg_post"].append(ndcg_at_k(relevances_post, k=5))

    print("\n================= Reranker 消融实验 =================")
    print(f"1. 召回率 (Recall@20 - 重排前): {np.mean(metrics['recall_pre']):.4f}")
    print(f"2. MRR@10 (重排前):             {np.mean(metrics['mrr_pre']):.4f}")
    print(f"3. MRR@5  (重排后):             {np.mean(metrics['mrr_post']):.4f}")
    print(f"4. NDCG@10(重排前):             {np.mean(metrics['ndcg_pre']):.4f}")
    print(f"5. NDCG@5 (重排后):             {np.mean(metrics['ndcg_post']):.4f}")


if __name__ == "__main__":
    run_ablation()