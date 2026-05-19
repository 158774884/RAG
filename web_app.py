import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs'))

import shutil
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from rag_engine import RAGEngine
from config import DOCUMENT_FOLDER, TOP_K, SERVER_PORT

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

@app.route('/api/list-dirs', methods=['POST'])
def api_list_dirs():
    try:
        data = request.get_json()
        dir_path = data.get('path', '').strip()

        if not dir_path:
            drives = []
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                d = letter + ':\\'
                if os.path.exists(d):
                    label = d
                    try:
                        import ctypes
                        buf = ctypes.create_unicode_buffer(128)
                        ctypes.windll.kernel32.GetVolumeInformationW(d, buf, 128, None, None, None, None, 0)
                        vol_name = buf.value
                        if vol_name:
                            label = f'{d} ({vol_name})'
                    except Exception:
                        pass
                    drives.append({'name': label, 'path': d.replace('\\', '/'), 'type': 'drive'})
            return jsonify({'status': 'ok', 'parent': None, 'current': '', 'items': drives})

        dir_path = os.path.normpath(dir_path)

        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            parent = None

        items = []
        try:
            entries = sorted(os.listdir(dir_path), key=lambda x: x.lower())
            for name in entries:
                full = os.path.join(dir_path, name)
                if os.path.isdir(full) and not name.startswith('.'):
                    items.append({
                        'name': name,
                        'path': full.replace('\\', '/'),
                        'type': 'dir'
                    })
        except PermissionError:
            pass

        return jsonify({
            'status': 'ok',
            'parent': parent.replace('\\', '/') if parent else None,
            'current': dir_path.replace('\\', '/'),
            'items': items
        })
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

        allowed_ext = {'.pdf', '.txt', '.docx', '.pptx', '.xlsx'}
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
    import argparse
    parser = argparse.ArgumentParser(description='RAG 知识库服务')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help=f'服务端口号 (默认: {SERVER_PORT})')
    args = parser.parse_args()

    PORT = args.port

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    if sock.connect_ex(('127.0.0.1', PORT)) == 0:
        sock.close()
        print(f"\n⚠ 端口 {PORT} 被占用，正在清理...")
        import subprocess, time
        result = subprocess.run(
            f'cmd /c "netstat -ano | findstr :{PORT}"',
            capture_output=True, text=True, shell=True
        )
        killed = set()
        for line in result.stdout.strip().split('\n'):
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if f':{PORT}' in p and i + 1 < len(parts):
                    pid = parts[-1]
                    if pid.isdigit() and pid not in killed:
                        killed.add(pid)
                        subprocess.run(f'cmd /c taskkill /F /PID {pid}',
                                       capture_output=True, shell=True)
                        print(f"  已终止 PID={pid}")
        time.sleep(1.5)
        print()
    else:
        sock.close()

    print(f"启动服务: http://127.0.0.1:{PORT}")
    print(f"配置文件端口: {SERVER_PORT}, 实际端口: {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)