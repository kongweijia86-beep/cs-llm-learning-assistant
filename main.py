from src.rag import RAG
rag = RAG("data/raw")
while True:
    question = input("\n请输入问题（exit 退出）：").strip()
    if not question:
        continue
    if question.lower() == "exit":
        break
    answer, contexts = rag.ask(question,top_k=top_k)
    print("\n================ AI回答 ================\n")
    print(answer)
    print("\n================ 参考资料 ================\n")
    for item in contexts:
        print(f"{item['source']} - Page {item['page']}  (score={item['score']:.4f})")