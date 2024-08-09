# This model implements WLAM (Weight Linear Additive Model) method
# It will also ask LLM for ranking advice
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
# import db_manager as db


class BlockRanker:
    # Define criteria
    criteria = ["web_score", "user_reputation", "block_score", "traffic_rate", "q_sim"]
    # Assign weight vectors (Different weight vectors for each source)
    weights = {
        "stackoverflow": [0.2, 0.1, 0.5, 0.1, 0.1],
        "reddit": [0.1, 0.1, 0.5, 0.2, 0.1],
        "codeproject": [0, 0.1, 0.5, 0, 0.4]
    }
    source_list = list(weights.keys())

    def __init__(self, data_list):
        self.__data_list = data_list
        self.__result = []
        self.normalize_traffic_rate()
        self.calculate_wlam_score()

    def normalize_traffic_rate(self):
        for s in self.source_list:
            curr_source = [d for d in self.__data_list if d['source'] == s]
            if len(curr_source) > 0:
                traffic_rate_list = [d['traffic_rate'] for d in curr_source]
                epsilon = 1e-8
                min_value = min(traffic_rate_list) + epsilon
                max_value = max(traffic_rate_list) + epsilon
                if min_value == max_value:
                    normalized_rate = [s / min_value for s in traffic_rate_list]
                else:
                    normalized_rate = [(x - min_value) / (max_value - min_value) for x in traffic_rate_list]
                for idx in range(len(normalized_rate)):
                    curr_source[idx]['traffic_rate'] = normalized_rate[idx]
                self.__result += curr_source
            else:
                continue

    # Weight Linear Additive Model
    def calculate_wlam_score(self):
        for alternative in self.__result:
            w = self.weights[alternative['source']]
            s = [alternative[c] for c in self.criteria]
            alternative['wlam_score'] = sum(w[idx] * s[idx] for idx in range(len(w)))

    def get_ranks(self):
        self.__result = sorted(self.__result, key=lambda x: x['wlam_score'], reverse=True)
        return self.__result[:20]


# Function to ask gpt to rank
class RankAssistant:
    GPT_MODEL = "gpt-3.5-turbo"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def __init__(self, ranks, q):
        self.__documents = ""
        for idx in range(len(ranks)):
            self.__documents += f"<<Document {idx}: , id: {ranks[idx]['id']}, " \
                                f"source: {ranks[idx]['source']}>>{ranks[idx]['content']}"

        self.messages = [
            {"role": "system", "content": "You are a professional programmer. "
                                          f"You will be provided with multiple unstructured documents about {q}, "
                                          "and your task is to optimize the rank of these documents."},
            {"role": "user", "content": f"My question is {q}. "
                                        f"Can you suggest a better rank for these documents?: {self.__documents}"}
        ]

        self.summary = [
            {"role": "system", "content": "You will be provided with multiple unstructured documents"
                                          " about programming issues. Your task is to summarize each document."},
            {"role": "user", "content": f"Please help me summarize these documents separately.{self.__documents}"}
        ]

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def rank_suggestion_request(self):
        try:
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=self.messages,
                temperature=0.7,
                top_p=1
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def doc_summary_request(self):
        try:
            response = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                messages=self.summary,
                temperature=0.7,
                top_p=1
            )
            return response
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            return e


if __name__ == "__main__":
    print("This is our block ranking method~")
    # test_data = db.query_previous_ranks("How can I create a stand-alone binary from a Python script?")[0]
    # print(test_data)
    # rank_assistant = RankAssistant(test_data['ranks'], test_data['question'])
    # response = rank_assistant.chat_completion_request()
    # print(response.choices[0].message.content)

