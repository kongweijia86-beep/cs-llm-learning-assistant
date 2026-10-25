import os
import json
import hashlib
import pymupdf

CACHE_DIR = 'cache'  ## 相对路径：随工作目录变化


class PDFLoader:  # 解析单个PDF
    def __init__(self):
        pass

    def load_pdf(self, pdf_path):
        source = os.path.basename(pdf_path)
        doc = pymupdf.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            pages.append({
                'text': page.get_text(),
                'page': i + 1,
                'source': source
            })
        doc.close()
        return pages


def load_all_pdfs(pdf_dir):
    os.makedirs(CACHE_DIR, exist_ok=True)
    filenames = sorted([
        f for f in os.listdir(pdf_dir) if f.lower().endswith('pdf')
    ])
    if not filenames:
        raise FileNotFoundError(f"{pdf_dir} 里没有找到pdf文件")

    # 同一目录、同一批文件 → 同一 key → 复用缓存
    key = hashlib.md5((pdf_dir + str(filenames)).encode()).hexdigest()
    # ★ 修正：把逗号改成点号 .json
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")

    # 读缓存
    if os.path.exists(cache_file):
        print("从缓存加载")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    loader = PDFLoader()
    all_pages = []
    for filename in filenames:
        pdf_path = os.path.join(pdf_dir, filename)
        print(f"正在读取{filename}")
        try:
            pages = loader.load_pdf(pdf_path)
            print(f"->{len(pages)}页")
            # ★ 核心修改：用 extend 而不是 append，把列表里的字典拍平
            all_pages.extend(pages)
        except Exception as e:
            print(f"跳过{filename}:{e}")
            continue

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    return all_pages