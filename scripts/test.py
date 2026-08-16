import requests
import json

url = "http://localhost:8000/api/v1/query/stream"
payload = {"question": "赵云长坂坡之战", "top_k": 3, "temperature": 0.3}

with requests.post(url, json=payload, stream=True) as response:
    full_answer = ""
    # 注意：这里不设置 decode_unicode=True，让 lines 保留为 bytes
    for line in response.iter_lines(decode_unicode=False):
        if not line:  # 跳过空行
            continue
        # 统一按字节处理，兼容 SSE 格式
        if line.startswith(b"data: "):
            data_str = line[6:].decode('utf-8')  # 去掉 "data: " 前缀并解码
            try:
                data = json.loads(data_str)
                if "token" in data:
                    print(data["token"], end="", flush=True)
                    full_answer += data["token"]
                elif "done" in data and data["done"]:
                    print("\n\n✅ 流式传输结束")
                    break
                elif "error" in data:
                    print(f"\n❌ 错误: {data['error']}")
                    break
            except json.JSONDecodeError:
                pass