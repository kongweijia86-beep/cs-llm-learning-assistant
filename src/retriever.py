# src/retriever.py

import numpy as np


class Retriever:
    """
    稠密检索器：基于 embedding 相似度从 chunks 里找最相关的文本块。
    """

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings   # np.ndarray, shape (N, D)，已归一化

    # ========== 单查询检索 ==========
    def search(self, query_embedding, top_k=3):
        """
        单查询稠密检索。
        返回 list[dict]，每个 dict 包含：
          - text:   chunk 文本
          - page:   页码
          - source: 来源 PDF
          - score:  余弦相似度（已归一化，等价于点积）
          - raw_score: 同 score，保留给 RRF 排查用
          - idx:    chunk 在 self.chunks 里的下标，供去重使用
        """
        scores = self.embeddings @ query_embedding  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            s = float(scores[idx])
            results.append({
                "text": self.chunks[idx]["text"],
                "page": self.chunks[idx]["page"],
                "source": self.chunks[idx]["source"],
                "score": s,
                "raw_score": s,
                "idx": idx,
            })
        return results

    # ========== Multi-Query 检索（保留，供兼容/单测用） ==========
    def search_multi_query(
        self,
        query_embeddings,          # list[np.ndarray]
        top_k=5,
        per_query_k=10,            # 每个子查询各取前多少个
        fusion="rrf",              # "rrf" 或 "max"
        rrf_k=60,                  # RRF 常数
    ):
        """
        多查询稠密检索 + 结果融合。
        注意：rag.py 的主流程已改成「逐路 search + 一次 RRF」，
        这个方法是保留给兼容旧调用或单独测试用。
        """
        per_query_results = []
        for q_emb in query_embeddings:
            res = self.search(q_emb, top_k=per_query_k)
            per_query_results.append(res)

        if fusion == "rrf":
            return self._rrf_fusion(per_query_results, top_k, rrf_k)
        elif fusion == "max":
            return self._max_fusion(per_query_results, top_k)
        else:
            raise ValueError(f"unknown fusion: {fusion}")

    # ========== RRF 融合（chunk 级，用 idx 去重） ==========
    def _rrf_fusion(self, per_query_results, top_k, rrf_k):
        """
        RRF：每个 chunk 得分 = sum(1 / (rrf_k + rank))
        - 用 idx 去重（同一 chunk 被多个子查询命中，RRF 分累加）
        - raw_score 记录该 chunk 见过的最高原始相似度（排查用）
        """
        scores = {}      # idx -> rrf score
        chunk_map = {}   # idx -> 该 chunk 的原始 dict
        best_raw = {}    # idx -> 该 chunk 见过的最高原始相似度

        for results in per_query_results:
            for rank, r in enumerate(results):
                idx = r["idx"]
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
                chunk_map[idx] = r
                best_raw[idx] = max(best_raw.get(idx, -1e9), r["score"])

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "source": chunk_map[idx]["source"],
                "page": chunk_map[idx]["page"],
                "score": float(scores[idx]),          # 融合后的 RRF 分
                "raw_score": float(best_raw[idx]),    # 原始相似度（排查用）
                "idx": idx,
            }
            for idx, _ in ranked
        ]

    # ========== Max 融合 ==========
    def _max_fusion(self, per_query_results, top_k):
        """
        Max：每个 chunk 取它见过的最高相似度
        """
        best = {}
        for results in per_query_results:
            for r in results:
                idx = r["idx"]
                if idx not in best or r["score"] > best[idx]["score"]:
                    best[idx] = r

        ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        return ranked

    # ========== 可选：页级融合（如果以后想在 retriever 层做） ==========
    def search_pages(self, query_embedding, top_k=5, per_page_chunks=1):
        """
        页级检索：把同一页的多个 chunk 分数取 max，按页返回。
        - per_page_chunks: 每页最多保留几个 chunk 的 text（用于拼接）
        """
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1]

        page_best = {}   # (source, page) -> (score, [texts])
        for idx in top_indices:
            idx = int(idx)
            c = self.chunks[idx]
            key = (c["source"], c["page"])
            s = float(scores[idx])
            if key not in page_best:
                page_best[key] = {"score": s, "texts": [c["text"]], "idx": idx}
            else:
                if s > page_best[key]["score"]:
                    page_best[key]["score"] = s
                    page_best[key]["idx"] = idx
                if len(page_best[key]["texts"]) < per_page_chunks:
                    page_best[key]["texts"].append(c["text"])

        ranked = sorted(page_best.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        results = []
        for (source, page), info in ranked:
            results.append({
                "source": source,
                "page": page,
                "text": "\n".join(info["texts"]),
                "score": info["score"],
                "raw_score": info["score"],
                "idx": info["idx"],
            })
        return results