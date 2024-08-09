import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import requests
import csv

tool_url = "http://127.0.0.1:8000/api/block_analysis"


def load_api(q):
    start = time.time()
    ranks = requests.post(url=tool_url, json={"q": q}, timeout=50).json()
    end = time.time()
    return {"rank": ranks, "time": end-start}


if __name__ == "__main__":
    with open("samples.json", "r", encoding="utf-8") as file:
        sample_q = json.load(file)
    file.close()

    category = list(sample_q.keys())
    # cat = category[0]
    all_questions = []
    for cat in category:
        for q in sample_q[cat]:
            print(q)
            response_time = []
            test_100_user = [q for i in range(99)]

            with ThreadPoolExecutor(max_workers=99) as executor:
                # Start the load operations and mark each future with its URL
                future_to_time = {executor.submit(load_api, q): q for q in test_100_user}

                # Record response time
                for future in as_completed(future_to_time):
                    r_time = future_to_time[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (r_time, exc))
                    else:
                        print(f'Response time: {data["time"]}')
                        response_time.append(format(data["time"], '.3f'))
            all_questions.append(response_time)

    with open("all_questions_response_time.csv", "w", encoding="utf-8") as file:
        csv_writer = csv.writer(file, delimiter=',')
        header_list = [f"User_{i}" for i in range(1,100)]
        csv_writer.writerow(header_list)
        for t_list in all_questions:
            csv_writer.writerow(t_list)


