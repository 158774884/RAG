import os, sys, traceback

log = open("log.txt", "w", encoding="utf-8")

def p(msg):
    print(msg)
    sys.stdout.flush()
    log.write(msg + "\n")
    log.flush()

try:
    p("Step 1: imports...")
    os.makedirs("./Doc", exist_ok=True)
    
    from rag_engine import RAGEngine
    from config import DOCUMENT_FOLDER
    p("  ok")

    test_file = os.path.join(DOCUMENT_FOLDER, "rag_intro.txt")
    if not os.path.exists(test_file):
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("RAG（Retrieval-Augmented Generation）检索增强生成是一种结合信息检索与文本生成的技术架构。\n\n")
            f.write("工作流程：首先将用户查询转换为向量，然后通过相似度搜索找到相关文档片段。\n")
            f.write("这些片段被注入到大语言模型的上下文提示词中，最终生成回答。\n\n")
            f.write("RAG优势：\n")
            f.write("1. 利用外部知识库，解决模型知识时效性问题\n")
            f.write("2. 提供可追溯的来源引用，增强回答可信度\n")
            f.write("3. 减少大模型产生幻觉的概率\n")

    p("Step 2: create engine...")
    rag = RAGEngine()
    p("  ok")

    p("Step 3: initialize...")
    rag.initialize(device="cpu")
    p("  ok")

    p("Step 4: build knowledge base...")
    result = rag.build_knowledge_base(DOCUMENT_FOLDER)
    if result["status"] == "success":
        p(f"  OK! docs={result['document_count']}, chunks={result['chunk_count']}")
    else:
        p(f"  info: {result['message']}")

    p("Step 5: stats...")
    stats = rag.get_db_stats()
    p(f"  chunks={stats['chunk_count']}")

    if stats["chunk_count"] > 0:
        p("Step 6: query test...")
        for q in ["什么是RAG？", "RAG有什么优势？"]:
            p(f"  Q: {q}")
            r = rag.query(q)
            if r["status"] == "success":
                p(f"  A: {r['answer'][:200]}")
            else:
                p(f"  fail: {r['message']}")

    p("ALL DONE")

except Exception as e:
    p(f"ERROR at step: {e}")
    p(traceback.format_exc())

log.close()
print("See log.txt")