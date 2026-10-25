import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(
    api_key = os.getenv('DEEPSEEK_API_KEY'),
    base_url = 'https://api.deepseek.com/v1'
)
def generate_answer(question,contexts):
    context_text = '\n\n'.join(
        [
            f"[{item['source']}-Page{item['page']}\n{item['text']}] " for item in contexts
        ]
    )
    prompt = f"""
资料：
{context_text}
用户问题：
{question}
要求：
1. 优先依据资料回答。
2. 如果资料中没有足够的信息，请明确告诉用户。
3. 不要编造资料中不存在的内容。
4. 尽量用容易理解的方式解释。
5. 最后列出使用的资料文件名和页码，格式：`文件名 - Page X`。
"""
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages = [{"role":'system',"content":f'你是 CS224N / NLP 学习助手。只根据用户提供的资料回答问题；资料不足时明确说明；不要编造'},
                     {"role":'user','content':f"{prompt}"}],
        temperature = 0.2
    )
    return response.choices[0].message.content