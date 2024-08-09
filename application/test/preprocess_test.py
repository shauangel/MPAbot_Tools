import json
from application.models import text_analysis_test as ta
import requests
from collections import Counter
import csv





q = "How to fix flask CORS error?"

data_controller_url = f"http://localhost:200/data" \
                      f"?keywords=python" \
                      f"&question={q}" \
                      f"&resources=stackoverflow;reddit;codeproject" \
                      f"&page=0&num=5"

resp = requests.get(url=data_controller_url, timeout=100).json()
analyzer = ta.TextAnalyze()
test_data = resp["result"]
# ranks = ta.block_ranking(post_items=test_data, question=q)
ans_corpus = [analyzer.batch_pipeline([ans['content'] for ans in post["answers"]]) for post in test_data]

pos_tag_list_list = []
token = 0
result = []
for ans_list in ans_corpus:
    for ans in ans_list:
        pos_tag_list_list.append(ans['pos_tag_list'])
        token += len(ans['result'])
        result.append({"pos_tag": ans['pos_tag_list']})

print(f"Total token: {token}")
with open("preprocessing/pos_tag_case0_compare.csv", 'w', encoding='utf-8') as file:
    fieldnames = list(result[0].keys())
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(result)

with open("preprocessing/pos_tag_case0_compare.json", 'w', encoding='utf-8') as file:
    json.dump(fp=file, obj=result)


