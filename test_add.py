import urllib.request, json

req = urllib.request.Request("http://127.0.0.1:5000/api/init", method="POST")
r = urllib.request.urlopen(req)
print("Init:", json.dumps(json.loads(r.read().decode()), ensure_ascii=False))

data = json.dumps({"paths": ["./Doc/rag_intro.txt"]}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:5000/api/add-files",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)
r = urllib.request.urlopen(req)
result = json.loads(r.read().decode())
print("Add #1:", json.dumps(result, ensure_ascii=False, indent=2))

r2 = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:5000/api/add-files",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
))
result2 = json.loads(r2.read().decode())
print("Add #2 (same file):", json.dumps(result2, ensure_ascii=False, indent=2))
