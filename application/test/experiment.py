import json
import time
import requests
from application.models import text_analysisV3 as ta

tool_url = "http://127.0.0.1:8000/api/block_analysis"

if __name__ == "__main__":

    with open("samples.json", "r", encoding="utf-8") as file:
        sample_q = json.load(file)
    file.close()

    category = list(sample_q.keys())
    cat = category[6]
    count = 1
    print(f"Category: {cat}")
    for q in sample_q[cat][2:]:
        print(f"Q{count}: {q}")
        count += 1
        start = time.time()
        data_controller_url = f"http://localhost:200/data" \
                              f"?keywords=python" \
                              f"&question={q}" \
                              f"&resources=stackoverflow;reddit;codeproject" \
                              f"&page=0&num=5"
        resp = requests.get(url=data_controller_url, timeout=100).json()
        end = time.time()
        for s in ["stackoverflow", "reddit", "codeproject"]:
            print(f">>> {s} response time: {resp[s]}")
        print(f"\nRetrieve time: {end-start}\n")

        # Count block amount
        # test_data = resp["result"]
        # analyzer = ta.TextAnalyze()
        # ans_corpus = [analyzer.batch_pipeline([ans['content'] for ans in post["answers"]]) for post in test_data]
        # corpus = sum([len(ans) for ans in ans_corpus])
        # print(f"Total Answer blocks: {corpus}")

        # Start Analyzing
        test_data = resp["result"]
        request = {"items": resp['result'],
                   "q": q}
        start = time.time()
        ranks = requests.post(url=tool_url, json=request, timeout=50)
        # analyzer = ta.TextAnalyze()
        # ranks = ta.block_ranking(post_items=test_data, question=q)
        for r in ranks:
            print(r)
        end = time.time()
        print(f"Final analyze time: {end-start}")

        print("---" * 15)

