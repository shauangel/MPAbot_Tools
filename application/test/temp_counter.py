import json
from collections import Counter
from application.models import text_analysis_test as ta


with open('old_test_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
analyzer = ta.TextAnalyze()
ans_corpus = [analyzer.batch_pipeline([ans['content'] for ans in post["answers"]]) for post in data]
token = 0
corpus = 0
unique_token = Counter()
for ans_list in ans_corpus:
    for ans in ans_list:
        corpus+=1
        token += len(ans['result'])
        unique_token+=Counter(ans['result'])

print(f"Token count :{token}")
print(f"Corpus count :{corpus}")
print(f"Unique Token: {len(unique_token.keys())}")