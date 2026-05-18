import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

import shutil
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine
from config import DOCUMENT_FOLDER, TOP_K

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag = RAGEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def api_init():
    try:
        rag.initialize(device='cpu')
        stats = rag.get_db_stats()
        return jsonify({'status': 'ok', 'message': '引擎初始化成功', 'chunks': stats['chunk_count']})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/build', methods=['POST'])
def api_build():
    try:
        if not rag.initialized:
            rag.initialize(device='cpu')
        result = rag.build_knowledge_base(DOCUMENT_FOLDER)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if not rag.initialized:
            rag.initialize(device='cpu')

        files = request.files.getlist('files')
        if not files:
            return jsonify({'status': 'error', 'message': '请选择文件'}), 400

        saved_paths = []
        for f in files:
            if f.filename:
                filename = secure_filename(f.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                f.save(save_path)
                saved_paths.append(save_path)

        if not saved_paths:
            return jsonify({'status': 'error', 'message': '没有有效文件'}), 400

        result = rag.add_documents(saved_paths)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        top_k = data.get('top_k', TOP_K)

        if not question:
            return jsonify({'status': 'error', 'message': '请输入问题'}), 400
        if not rag.initialized:
            rag.initialize(device='cpu')

        result = rag.query(question, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    try:
        if not rag.initialized:
            return jsonify({'status': 'ok', 'initialized': False, 'document_count': 0, 'chunk_count': 0})
        stats = rag.get_db_stats()
        stats['initialized'] = True
        return jsonify(stats)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def api_clear():
    try:
        if rag.initialized:
            rag.clear_knowledge_base()
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        return jsonify({'status': 'ok', 'message': '知识库已清空'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scan-dir', methods=['POST'])
def api_scan_dir():
    try:
        data = request.get_json()
        dir_path = data.get('path', '').strip()
        if not dir_path:
            return jsonify({'status': 'error', 'message': '请输入目录路径'}), 400
        if not os.path.isdir(dir_path):
            return jsonify({'status': 'error', 'message': '目录不存在'}), 400

        allowed_ext = {'.pdf', '.txt', '.docx'}
        files = []
        for root, dirs, filenames in os.walk(dir_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in allowed_ext:
                    full_path = os.path.join(root, fname)
                    size_kb = round(os.path.getsize(full_path) / 1024, 1)
                    files.append({
                        'name': fname,
                        'path': full_path.replace('\\', '/'),
                        'folder': root.replace('\\', '/'),
                        'size_kb': size_kb,
                        'ext': ext[1:].upper()
                    })
        return jsonify({'status': 'ok', 'files': files, 'count': len(files)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/add-files', methods=['POST'])
def api_add_files():
    try:
        if not rag.initialized:
            rag.initialize(device='cpu')

        data = request.get_json()
        paths = data.get('paths', [])
        if not paths:
            return jsonify({'status': 'error', 'message': '请选择文件'}), 400

        valid_paths = [p for p in paths if os.path.isfile(p)]
        if not valid_paths:
            return jsonify({'status': 'error', 'message': '没有有效文件'}), 400

        result = rag.add_documents(valid_paths)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/added-files', methods=['GET'])
def api_added_files():
    try:
        if not rag.initialized:
            rag.initialize(device='cpu')
        files = rag.get_added_files()
        for f in files:
            f['name'] = os.path.basename(f['source'])
            f['folder'] = os.path.dirname(f['source']).replace('\\', '/')
            f['size_kb'] = round(f.get('bytes', 0) / 1024, 1)
            ext = os.path.splitext(f['source'])[1].lower()
            f['ext'] = ext[1:].upper() if ext else '?'
        return jsonify({'status': 'ok', 'files': files, 'count': len(files)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/remove-file', methods=['POST'])
def api_remove_file():
    try:
        if not rag.initialized:
            rag.initialize(device='cpu')
        data = request.get_json()
        path = data.get('path', '').strip()
        if not path:
            return jsonify({'status': 'error', 'message': '请指定文件路径'}), 400
        result = rag.remove_document(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)