from locust import task, HttpUser
from locust_sse import SSEUser
import random
import time
from questions import QUESTION_POOL

class RAGStreamUser(HttpUser):

    url = "http://localhost:8000"
    count = 0

    @task
    def query_stream(self):
        payload = {
            "question": random.choice(QUESTION_POOL),
            "top_k": 3,
            "temperature": 0.3,
            "stream": False
        }

        start_time = time.time()
        request_name = "rag-quest-stream"

        with self.client.post(
            "/api/v1/query/stream",
            json=payload,
            catch_response=True,
            stream=True,
            name=request_name,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

            total_time = int((time.time() - start_time) * 1000)
            self.count += 1
            print(f"第{self.count }次请求 | 耗时：total_time: {total_time}\n")
            response.success()

    