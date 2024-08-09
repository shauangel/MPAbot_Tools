import json
import time
from application.models import text_analysisV3 as ta


if __name__ == "__main__":

    with open("samples.json", 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    start = time.time()
    # extract keywords from user question
    analyzer = ta.TextAnalyze()
    # qkey, doc = analyzer.content_pre_process(i['question'])
    # for i in range(len(test_data["result"])):
    #     test_batch = [data["content"] for data in test_data["result"][i]["answers"]]
    # qkey = analyzer.content_pre_process(test_batch[0])
    # print(f"<<Original Content>>\n{test_batch[0]}")
    # print("---"*10)
    # print(f"<<Preprocessed Tokens>>\n{qkey}")

        # Test batch process
    #     prec_pipe = analyzer.batch_pipeline(test_batch)
    #     print(prec_pipe)

    rank = ta.block_ranking(post_items=test_data['result'], question="How to fix flask cors error?")
    end = time.time()
    for r in rank:
        print(r)

    print("Process time: " + str(end-start))
