import time
import requests
from application.models import text_analysisV3 as ta
from application.models.block_ranker import RankAssistant
from application.models import db_manager as db

q = "How to fix flask CORS error?"
tool_url = "http://127.0.0.1:8000/api/block_analysis"


# Search Question & Retrieve Data
print(f"Demo : {q}")
start = time.time()
data_controller_url = f"http://localhost:200/data" \
                      f"?keywords=python" \
                      f"&question={q}" \
                      f"&resources=stackoverflow;reddit;codeproject" \
                      f"&page=0&num=10"
resp = requests.get(url=data_controller_url, timeout=100).json()
end = time.time()
print(f"Retrieve time: {end - start}")

# Start Analyzing
test_data = resp["result"]
request = {"items": resp['result'],
           "q": q}
start = time.time()
# ranks = requests.post(url=tool_url, json=request, timeout=50)
analyzer = ta.TextAnalyze()
ranks = ta.block_ranking(post_items=test_data, question=q)
for r in ranks:
    print(r)
end = time.time()
print(f"Analysis time: {end - start}")
print("---" * 10)

"""
test_data = db.query_previous_ranks(q)[0]
# print(test_data)
rank_assistant = RankAssistant(test_data['ranks'], test_data['question'])
# rank by LLM
response = rank_assistant.rank_suggestion_request()
print(response.choices[0].message.content)
with open(f"LLM_responses/ranks/Demo.txt", 'w', encoding='utf-8') as textfile:
    textfile.write(f"Question: {q}\n")
    textfile.write(response.choices[0].message.content)
textfile.close()

s_response = rank_assistant.doc_summary_request()
# rank by MPAbot, summarized by LLM
with open(f"LLM_responses/summary/Demo.txt", 'w', encoding='utf-8') as file:
    file.write(f"Question: {q}\n")
    file.write(s_response.choices[0].message.content)
"""