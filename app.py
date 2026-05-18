import os
from rag_engine import RAGEngine
from config import DOCUMENT_FOLDER, VECTOR_DB_PATH

rag_engine = RAGEngine()
initialized = False

def initialize_rag(device: str = "cpu") -> str:
    global initialized
    try:
        rag_engine.initialize(device=device)
        initialized = True
        stats = rag_engine.get_db_stats()
        return f"✅ RAG引擎初始化成功！\n知识库状态: {stats['chunk_count']} 个文档片段"
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"

def build_knowledge_base() -> str:
    if not initialized:
        return "请先初始化RAG引擎"
    
    try:
        result = rag_engine.build_knowledge_base(DOCUMENT_FOLDER)
        if result["status"] == "success":
            return f"✅ 知识库构建成功！\n加载文档: {result['document_count']} 个\n生成片段: {result['chunk_count']} 个"
        else:
            return f"❌ {result['message']}"
    except Exception as e:
        return f"❌ 构建失败: {str(e)}"

def add_documents(files) -> str:
    if not initialized:
        return "请先初始化RAG引擎"
    
    if not files:
        return "请选择要上传的文件"
    
    try:
        file_paths = [file.name for file in files]
        result = rag_engine.add_documents(file_paths)
        return f"✅ 文档添加成功！\n新增片段: {result['chunk_count']} 个"
    except Exception as e:
        return f"❌ 添加失败: {str(e)}"

def query_rag(question: str, top_k: int) -> str:
    if not initialized:
        return "请先初始化RAG引擎"
    
    if not question.strip():
        return "请输入问题"
    
    try:
        result = rag_engine.query(question, top_k=top_k)
        if result["status"] == "error":
            return f"❌ {result['message']}"
        
        answer = result["answer"]
        sources = result["sources"]
        
        response = f"🤖 回答：\n{answer}\n\n"
        if sources:
            response += "📄 参考来源：\n"
            for i, source in enumerate(sources[:3], 1):
                response += f"\n{i}. 文件: {os.path.basename(source['source'])}\n内容: {source['content'][:100]}..."
        
        return response
    except Exception as e:
        return f"❌ 查询失败: {str(e)}"

def clear_knowledge() -> str:
    if not initialized:
        return "请先初始化RAG引擎"
    
    try:
        result = rag_engine.clear_knowledge_base()
        return f"✅ {result['message']}"
    except Exception as e:
        return f"❌ 清空失败: {str(e)}"

def get_current_stats() -> str:
    if not initialized:
        return "请先初始化RAG引擎"
    
    stats = rag_engine.get_db_stats()
    return f"📊 当前知识库统计:\n文档片段数: {stats['chunk_count']}"

def main():
    try:
        import gradio as gr
        
        with gr.Blocks(title="RAG知识库搭建工具", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 📚 RAG知识库搭建工具")
            gr.Markdown("基于LangChain和HuggingFace的本地RAG知识库系统")
            
            with gr.Tab("初始化"):
                device_choice = gr.Radio(["cpu", "gpu"], label="设备选择", value="cpu")
                init_btn = gr.Button("初始化RAG引擎", variant="primary")
                init_output = gr.Textbox(label="初始化状态", lines=3)
                init_btn.click(initialize_rag, inputs=device_choice, outputs=init_output)
                
                stats_btn = gr.Button("查看知识库状态")
                stats_output = gr.Textbox(label="知识库统计", lines=2)
                stats_btn.click(get_current_stats, outputs=stats_output)
            
            with gr.Tab("文档管理"):
                gr.Markdown("### 📁 从文件夹构建知识库")
                build_btn = gr.Button("构建知识库", variant="primary")
                build_output = gr.Textbox(label="构建结果", lines=3)
                build_btn.click(build_knowledge_base, outputs=build_output)
                
                gr.Markdown("### 📤 上传文档")
                file_upload = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".docx"])
                upload_btn = gr.Button("添加文档", variant="secondary")
                upload_output = gr.Textbox(label="上传结果", lines=2)
                upload_btn.click(add_documents, inputs=file_upload, outputs=upload_output)
                
                gr.Markdown("### 🗑️ 清空知识库")
                clear_btn = gr.Button("清空知识库", variant="stop")
                clear_output = gr.Textbox(label="清空结果", lines=2)
                clear_btn.click(clear_knowledge, outputs=clear_output)
            
            with gr.Tab("问答"):
                gr.Markdown("### 💬 向知识库提问")
                question_input = gr.Textbox(label="输入问题", placeholder="请输入您的问题...", lines=3)
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, label="检索数量")
                query_btn = gr.Button("提问", variant="primary")
                query_output = gr.Textbox(label="回答结果", lines=10)
                query_btn.click(query_rag, inputs=[question_input, top_k_slider], outputs=query_output)
            
            with gr.Tab("使用说明"):
                gr.Markdown("""
                ## 📖 使用说明
                
                ### 第一步：初始化引擎
                1. 在「初始化」标签页选择运行设备（CPU或GPU）
                2. 点击「初始化RAG引擎」按钮
                
                ### 第二步：构建知识库
                方法一：从文件夹构建
                - 将PDF/TXT/DOCX文档放入 `Doc` 文件夹
                - 在「文档管理」标签页点击「构建知识库」
                
                方法二：上传文档
                - 在「文档管理」标签页上传文档
                - 点击「添加文档」按钮
                
                ### 第三步：开始问答
                - 在「问答」标签页输入问题
                - 点击「提问」按钮获取答案
                
                ### 支持的文档格式
                - PDF文件 (.pdf)
                - 文本文件 (.txt)
                - Word文档 (.docx)
                
                ### 技术栈
                - LangChain: 框架
                - Chroma: 向量数据库
                - BGE: 中文Embedding模型
                - HuggingFace: 语言模型
                """)
        
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
    except ImportError:
        print("⚠️ Gradio未安装，启动命令行模式")
        print("=" * 50)
        
        print("\n1. 初始化RAG引擎...")
        rag_engine.initialize(device="cpu")
        print("✅ RAG引擎初始化成功")
        
        stats = rag_engine.get_db_stats()
        print(f"当前知识库: {stats['chunk_count']} 个文档片段")
        
        while True:
            print("\n" + "=" * 50)
            print("请选择操作:")
            print("1. 构建知识库")
            print("2. 向知识库提问")
            print("3. 查看知识库状态")
            print("4. 清空知识库")
            print("5. 退出")
            
            choice = input("\n输入选择 (1-5): ")
            
            if choice == "1":
                print("\n构建知识库中...")
                result = rag_engine.build_knowledge_base(DOCUMENT_FOLDER)
                if result["status"] == "success":
                    print(f"✅ 构建成功！")
                    print(f"  - 加载文档: {result['document_count']} 个")
                    print(f"  - 生成片段: {result['chunk_count']} 个")
                else:
                    print(f"❌ 构建失败: {result['message']}")
            
            elif choice == "2":
                if rag_engine.db is None:
                    print("❌ 知识库尚未构建，请先选择选项1")
                    continue
                
                question = input("\n请输入问题: ")
                if not question.strip():
                    print("请输入有效问题")
                    continue
                
                print("正在思考...")
                result = rag_engine.query(question, top_k=5)
                
                if result["status"] == "success":
                    print("\n🤖 AI回答:")
                    print(result["answer"])
                    
                    if result["sources"]:
                        print("\n📄 参考来源:")
                        for i, source in enumerate(result["sources"][:3], 1):
                            print(f"\n{i}. 文件: {source['source']}")
                            print(f"   内容: {source['content'][:100]}...")
                else:
                    print(f"❌ 查询失败: {result['message']}")
            
            elif choice == "3":
                stats = rag_engine.get_db_stats()
                print(f"\n📊 知识库统计:")
                print(f"  - 文档片段数: {stats['chunk_count']}")
            
            elif choice == "4":
                confirm = input("确定要清空知识库吗? (y/N): ")
                if confirm.lower() == "y":
                    result = rag_engine.clear_knowledge_base()
                    print(f"✅ {result['message']}")
                else:
                    print("取消操作")
            
            elif choice == "5":
                print("\n👋 再见！")
                break
            
            else:
                print("❌ 无效选择，请输入1-5")

if __name__ == "__main__":
    main()