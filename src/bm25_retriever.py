from rank_bm25 import BM25Okapi
from src.tokenizer import tokenize
class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        print("正在构建 BM25 索引...")
        tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    def search(self, query, top_k=5):
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 获取 top_k 的索引
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "page": self.chunks[idx]["page"],
                "source": self.chunks[idx]["source"],
                "score": float(scores[idx]),
            })
        return results
