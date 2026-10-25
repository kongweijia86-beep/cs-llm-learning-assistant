import os
import re
from pathlib import Path
from collections import defaultdict
from .pdf_loader import load_all_pdfs
from .chunker import chunk_pages
from .retriever import Retriever
from .page_filter import(filter_retrieval_results,diversify_by_chapter)
from .bm25_retriever import BM25Retriever
try:
    from dotenv import load_dotenv
    def _find_and_load_env():
        p = Path(__file__).resolve().parent
        for _ in range(5):
            env_file = p/'.env'
            if env_file.exists():
                load_dotenv(env_file)
                print(f"[rag] 已加载 .env: {env_file}")
                return
            p = p.parent
        print("[rag] 未找到 .env")
    _find_and_load_env()
except ImportError:
    pass
def get_max_per_chapter(question_item):
    if not question_item:
        return 3
    chapter = question_item.get('chapter','') or ""
    if "&" in chapter:
        return 3
    return 3
class RAG:
    def __init__(self,pdf_dir,chunk_size=500,chunk_overlap=50,use_reranker=True,reranker_model = 'BAAI/bge-reranker-v2-m3'):
        from sentence_transformers import SentenceTransformer
        self.pages = load_all_pdfs(pdf_dir)
        self.chunks = chunk_pages(self.pages, chunk_size, chunk_overlap)
        print(f"生成 {len(self.chunks)} 个 chunks")
        print('正在生成Embedding')
        self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        texts = [c['text'] for c in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.retriever = Retriever(self.chunks,self.embeddings)
        print("正在构建BM25索引")
        self.bm25_retriever = BM25Retriever(self.chunks)
        self.use_reranker = use_reranker
        if use_reranker:
            from .reranker import Reranker
            self.reranker = Reranker(model_name=reranker_model)
        else:
            self.reranker = None
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            from openai import OpenAI
            self.llm_client = OpenAI(
                api_key = api_key,
                base_url = "https://api.deepseek.com",
            )
            print(f"[rag] DeepSeek 客户端已初始化（key: {api_key[:10]}...）")
        else:
            print("[rag] DEEPSEEK_API_KEY 未设置，Multi-Query 将回退到单查询")
            self.llm_client = None
        self._page_text_index = defaultdict(list)
        for c in self.chunks:
            self._page_text_index[(c['source'],c['page'])].append(c['text'])
    def _compute_recall_k(self, top_k,question_item):#给定最终要返回的 top_k 条，算出粗召回该捞多少条 recall_k
        recall_k = top_k *4 if question_item is not None else top_k ## 有附加信息 → 放大 4 倍
        if self.use_reranker:
            recall_k = max(recall_k,top_k*6)
        return recall_k
    def _rrf_fuse(self,*result_lists,k=60):
        rrf_scores = {} #记录每个页的RRF累计分数
        doc_map = {} #记录每个页的代表文档内容（只存第一次遇到的)
        page_texts = defaultdict(list)
        for results in result_lists:
            for rank,doc in enumerate(results,start=1):
                key = (doc['source'],doc['page']) #构造去重key
                rrf_scores[key] = rrf_scores.get(key,0.0) + 1.0/(k+rank)
                if key not in doc_map:
                    doc_map[key] = dict(doc)
                if doc.get('text'):
                    page_texts[key].append(doc['text'])
        sorted_keys = sorted(rrf_scores.keys(),key = lambda kk:rrf_scores[kk],reverse=True)
        merged = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc['score'] = rrf_scores[key]
            texts = page_texts.get(key,[])
            if texts:
                seen = set() # 判重工具
                uniq = [] #存放去重结果
                for t in texts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                doc['text'] = '\n'.join(uniq)
            merged.append(doc)
        return merged
    #补齐text
    def _ensure_text(self,results):
        out = []
        for r in results:
            item = dict(r)
            if 'text' not in item or not item['text']:
                key = (item['source'],item['page'])
                texts = self._page_text_index.get(key,[])
                item['text'] = '\n'.join(texts) if texts else ''
            out.append(item)
        return out
    #单查询入口
    def search(self,query_emb,query_text=None,top_k=5,question_item=None,tolerance=2,use_bm25=True):
        recall_k = self._compute_recall_k(top_k,question_item)
        #稠密搜索
        dense_results = self.retriever.search(query_emb,top_k=recall_k)
        print("Dense top-20 pages:",
              [(r["page"], round(r["score"], 3)) for r in dense_results[:20]])
        #BM25检索
        if use_bm25 and query_text:
            bm25_results = self.bm25_retriever.search(query_text,top_k=recall_k)
            print("BM25 top-20 pages:",
                  [(r["page"], round(r["score"], 3)) for r in bm25_results[:20]])
            merged = self._rrf_fuse(dense_results, bm25_results)
        else:
            merged = dense_results
        if question_item is not None:
            merged = diversify_by_chapter(
                merged,
                max_per_chapter=get_max_per_chapter(question_item),
            )
            print(f"  [DEBUG search] 融合后 {[r['page'] for r in merged[:top_k]]}")
            merged = filter_retrieval_results(merged,question_item,tolerance=tolerance)
            print(f"  [DEBUG search] 过滤后 {[r['page'] for r in merged[:top_k]]}")
        #rerank
        if self.use_reranker and query_text and merged:
            merged = self._ensure_text(merged)
            merged = self.reranker.rerank(query_text,merged,top_k=top_k)
            print(f"  [DEBUG search] rerank后 {[r['page'] for r in merged[:top_k]]}")
        return merged[:top_k]

    def rewrite_query(self, query, n_queries=3):
        if self.llm_client is None:
            return [query]
        prompt = f"""你是一个查询拆分助手。请把下面的**复杂问题**拆成 {n_queries} 个**独立子问题**，每个子问题聚焦原问题的一个**子话题**。
    保留原文题里的章节信息
    【示例 1】
    原问题：Compare classification-finetuning and instruction-finetuning in terms of required data, compute, and range of tasks.
    子问题：
    What data is required for classification-finetuning?
    What data is required for instruction-finetuning?
    What compute and tasks are involved in instruction-finetuning compared to classification-finetuning?

    【示例 2】
    原问题：How does the computation graph concept from Appendix A help explain backpropagation through self-attention in Chapter 3?
    子问题：
    What is a computational graph in Appendix A?
    How does backpropagation work through the self-attention mechanism in Chapter 3?
    How does PyTorch build the computation graph for self-attention and compute gradients?

    【示例 3】
    原问题：How do the data preparation and batching approaches differ between classification-finetuning in Chapter 6 and instruction-finetuning in Chapter 7?
    子问题：
    How is data prepared and batched in classification-finetuning in Chapter 6?
    How is data prepared and batched in instruction-finetuning in Chapter 7?
    What are the key differences in data preparation between the two chapters?

    【要求】
    1. **每个子问题聚焦一个独立的语义点，不要用同义词改写原问题**
    2. 子问题之间不要语义重叠
    3. 子问题保持和原问题相同的语言
    4. 只输出子问题，每行一个，不要编号，不要解释
    6. 如果原问题里包含章节号（如 Chapter 3、Appendix A），子问题必须保留相同的章节号。
    7. 不要用同义词替换专有名词（如 TransformerBlock、CausalAttention 要保持原样）。
    原问题：{query}

    子问题："""
        try:
            response = self.llm_client.chat.completions.create(
                model = 'deepseek-chat',
                messages = [{
                    'role':'user','content':prompt
                }],
                temperature=0.3,
                max_tokens = 300
            )
            text = response.choices[0].message.content.strip()
            sub_queries = [ln.strip() for ln in text.split("\n") if ln.strip()]
            sub_queries = [re.sub(r'^\s*[\d]+[\.\)]\s*', '', q) for q in sub_queries]
            sub_queries = [re.sub(r'^\s*[-•]\s*', '', q) for q in sub_queries]
            sub_queries = [q for q in sub_queries if q]
        except Exception as e:
            print(f"[rewrite_query] DeepSeek 调用失败: {e}，回退到单查询")
            sub_queries = []
        return [query] + sub_queries
    def search_multi_query(self, query, top_k=5, n_queries=3,
                           per_query_k=10, question_item=None,
                           tolerance=2, use_bm25=True):
        queries = self.rewrite_query(query,n_queries=n_queries)
        print(f"[multi-query] 原问题 + {len(queries)-1}个子问题")
        for i,q in enumerate(queries):
            print(f"    Q{i}: {q}")
        q_embs = self.embedding_model.encode(
            queries,normalize_embeddings=True
        )
        recall_k = self._compute_recall_k(top_k, question_item)
        # 收集所有路：稠密每一路 + BM25 每一路
        all_lists = []
        for q_emb in q_embs:
            dense_hits = self.retriever.search(q_emb,top_k=recall_k)
            all_lists.append(dense_hits)
        if use_bm25:
            for q in queries:
                bm25_hits = self.bm25_retriever.search(q, top_k=recall_k)
                all_lists.append(bm25_hits)
        merged = self._rrf_fuse(*all_lists)
        if question_item is not None:
            merged = diversify_by_chapter(
                merged,
                max_per_chapter=get_max_per_chapter(question_item),
            )
            print(f"  [DEBUG mq] 融合后 {[r['page'] for r in merged[:top_k]]}")
            merged = filter_retrieval_results(
                merged, question_item, tolerance=tolerance,
            )
            print(f"  [DEBUG mq] 过滤后 {[r['page'] for r in merged[:top_k]]}")
        if self.use_reranker and merged:
            merged = self._ensure_text(merged)
            merged = self.reranker.rerank(query,merged,top_k=top_k)
            print(f"  [DEBUG mq] rerank后 {[r['page'] for r in merged[:top_k]]}")
        return merged[:top_k]
    def ask(self,question,top_k=5,category=None,chapter=None):
        from eval.generate_eval import generate_answer
        if self.llm_client is None:
            return "LLM 客户端未初始化，无法生成答案。", []
        question_item = {
            "question": question,
            "category": category,
            "chapter": chapter,
        }
        if category in {'cross_chapter','multihop'}:
            contexts = self.search_multi_query(
                question, top_k=top_k, n_queries=3, per_query_k=10,
                question_item=question_item, tolerance=1,
            )
        else:
            query_emb = self.embedding_model.encode([question])[0]
            contexts = self.search(
                query_emb, query_text=question, top_k=top_k,
                question_item=question_item, tolerance=1,
            )
        answer = generate_answer(question, contexts, self.llm_client)
        return answer, contexts

