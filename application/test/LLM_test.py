import json
import time
import requests
from application.models.block_ranker import RankAssistant
from application.models import db_manager as db


if __name__ == "__main__":

    with open("samples.json", "r", encoding="utf-8") as file:
        sample_q = json.load(file)
    file.close()
    count = 1
    for cat in sample_q[:1]:
        print(f"Category: {cat}")
        for q in sample_q[cat][:1]:
            print(f"Question {count}: {q}")
            test_data = db.query_previous_ranks(q)[0]
            # print(test_data)
            rank_assistant = RankAssistant(test_data['ranks'], test_data['question'])
            # rank by LLM
            response = rank_assistant.rank_suggestion_request()
            print(response.choices[0].message.content)
            with open(f"LLM_responses/ranks/Q{count}.txt", 'w', encoding='utf-8') as textfile:
                textfile.write(f"Question: {q}\n")
                textfile.write(response.choices[0].message.content)
            textfile.close()

            s_response = rank_assistant.doc_summary_request()
            # rank by MPAbot, summarized by LLM
            with open(f"LLM_responses/summary/Q{count}.txt", 'w', encoding='utf-8') as file:
                file.write(f"Question: {q}\n")
                file.write(s_response.choices[0].message.content)
            count += 1


