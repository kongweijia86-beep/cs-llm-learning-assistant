import re
def tokenize(text):
    """
    英文分词，用于 BM25：
    - 转小写
    - 连字符替换成空格（self-attention → self attention）
    - 去掉标点
    - 按空格切分
    """
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()