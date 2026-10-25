import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    def __init__(self):
        print("加载embedding模型")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    def encode(self,texts):
        return self.model.encode(
            texts,
            normalize_embeddings = True, #向量归一化
            show_progress_bar = True #进度条显示
        )
