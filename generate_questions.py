import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = 'https://api.deepseek.com/v1'
)
def generate_for_chunk(chunk_text,n,chunk_size):
    prompt = f"""下面是书 从零建立大模型 的内容。
    请根据这段内容，生成 {n} 个高质量的评测问题，要求：
    1. 用英文提问（资料是英文）
    2. 每题针对一个明确概念，答案必须能在这段内容里找到
    3. 难度和类型多样：概念、公式、对比、原理
    4. 每题给出参考答案
    只输出 JSON 数组，格式：
    [
      {{"question": "...", "reference_answer": "...", "category": "concept"}},
      ...
    ]
    资料：
    {chunk_text[:chunk_size]}
    """
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages = [{'role':'system','content':"你是一名熟悉 NLP 和大模型的课程助教，负责根据教材内容生成高质量的评测题。"},{'role':'user','content':prompt}],
        temperature = 0.7
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception as e:
        print(f'解析失败: {e}\n原文前200字: {text[:200]}')
        return []
def main():
    from src.pdf_loader import load_all_pdfs
    from src.chunker import chunk_pages
    pages = load_all_pdfs('data/raw')
    print(f"读取到{len(pages)}个页面")
    chunks = chunk_pages(
        pages,chunk_size=8000,overlap=500
    )
    print(f"生成{len(chunks)}个chunks")
    candidates = []
    for i,chunk in enumerate(chunks):
        print(f"正在处理chunk"
              f"{i+1}/{len(chunks)}")
        questions = generate_for_chunk(chunk['text'],n=3,chunk_size=8000)
        for q in questions:
            q['source'] = chunk['source']
            q['page_hint'] = chunk['page']
            q['chunk_id'] = i
            candidates.append(q)
    os.makedirs('eval',exist_ok=True)
    output_path = 'eval/candidates.json'
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(candidates,f,ensure_ascii=False,indent=2)
        print("\n======================")
        print("评测题生成完成")
        print("======================")
        print(f"Chunk 数量：{len(chunks)}")
        print(f"候选题数量：{len(candidates)}")
        print(f"保存位置：{output_path}")
if __name__ == "__main__":
    main()

