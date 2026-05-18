import os
import sys
from rag_engine import RAGEngine
from config import DOCUMENT_FOLDER

def main():
    print("\n" + "=" * 60)
    print("  RAG 知识库搭建工具 - 命令行版")
    print("=" * 60)

    rag = RAGEngine()

    print("\n初始化引擎...")
    try:
        rag.initialize(device="cpu")
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    while True:
        print("\n" + "=" * 60)
        print("  1. 构建知识库（扫描 Doc 文件夹）")
        print("  2. 向知识库提问")
        print("  3. 查看知识库状态")
        print("  4. 清空知识库")
        print("  5. 退出")
        print("=" * 60)

        choice = input("\n请输入选项 (1-5): ").strip()

        if choice == "1":
            print("\n正在扫描文档...")
            result = rag.build_knowledge_base(DOCUMENT_FOLDER)
            if result["status"] == "success":
                print(f"构建成功！文档: {result['document_count']}, 片段: {result['chunk_count']}")
            else:
                print(f"构建失败: {result['message']}")

        elif choice == "2":
            question = input("\n请输入问题: ").strip()
            if not question:
                continue
            print("检索中...")
            result = rag.query(question)
            if result["status"] == "success":
                print("\n" + "-" * 40)
                print("回答:")
                print(result["answer"])
                print("-" * 40)
                if result["sources"]:
                    print("\n参考来源:")
                    for i, s in enumerate(result["sources"][:3], 1):
                        print(f"  [{i}] {os.path.basename(s['source'])}")
            else:
                print(f"查询失败: {result['message']}")

        elif choice == "3":
            stats = rag.get_db_stats()
            print(f"\n片段数: {stats['chunk_count']}")

        elif choice == "4":
            if input("确认清空? (y/N): ").strip().lower() == "y":
                print(rag.clear_knowledge_base()["message"])
            else:
                print("已取消")

        elif choice == "5":
            print("\n再见！")
            break
        else:
            print("无效选项")

if __name__ == "__main__":
    main()