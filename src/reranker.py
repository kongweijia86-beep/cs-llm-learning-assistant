class Reranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        from sentence_transformers import CrossEncoder
        print(f"正在加载Reranker:{model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Reranker加载完成")

    def rerank(self, query, candidates, top_k=5):
        if not candidates:
            return []
        pairs = [(query, c['text']) for c in candidates]
        scores = self.model.predict(pairs)
        scored = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item['rerank_score'] = float(s)
            scored.append(item)
        scored.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored[:top_k]