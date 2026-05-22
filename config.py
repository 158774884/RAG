import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DOCUMENT_FOLDER = "./Doc"
VECTOR_DB_PATH = "./chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5

SERVER_PORT = 5000

LLM_ENABLED = False
LLM_TYPE = "online"
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = ""
LLM_BASE_URL = "https://api.deepseek.com"
LLM_LOCAL_URL = "http://localhost:11434/api/generate"
LLM_LOCAL_MODEL = "qwen2.5:7b"

LLM_CONFIG_FILE = "./llm_config.json"

LLM_PRESETS = [
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com", "model": "deepseek-chat"},
    {"name": "通义千问 (Qwen)", "base_url": "https://dashscope.aliyuncs.com/compatible-mode", "model": "qwen-turbo"},
    {"name": "豆包 (Doubao)", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "ep-2024"},
    {"name": "智谱 (GLM)", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    {"name": "Moonshot (Kimi)", "base_url": "https://api.moonshot.cn", "model": "moonshot-v1-8k"},
    {"name": "零一万物 (Yi)", "base_url": "https://api.lingyiwanwu.com", "model": "yi-large"},
    {"name": "OpenAI", "base_url": "https://api.openai.com", "model": "gpt-4o"},
    {"name": "自定义", "base_url": "", "model": ""},
]