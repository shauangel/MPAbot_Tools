from flask import request, Blueprint, jsonify, render_template
from models import text_analysisV0, text_analysisV2, text_analysisV1, text_analysisV3
from models import db_manager as db
import json
import requests

# register api
text_analysis_api = Blueprint("text_analysis_api", __name__)

# data controller settings
data_controller_url = "http://localhost:200/data"


# Latest block analysis method, more Criteria considered, faster analyzing
@text_analysis_api.route("/block_analysis", methods=['POST'])
def block_analysis():
    data = request.get_json()
    try:
        print("Processing Block Analysis V.3")
        # Step 1. Check related
        query = db.query_previous_ranks(data['q'])
        try:
            print("Retrieve history")
            ranks = query[0]['ranks']
        except IndexError:
            # Step 1: Request data
            url = data_controller_url + f"?keywords=python" \
                                        f"&question={data['q']}" \
                                        f"&resources=stackoverflow;reddit;codeproject" \
                                        f"&page=0&num=10"
            resp = requests.get(url=url, timeout=100).json()
            # Step 2: Start Analyzing
            data = resp["result"]
            ranks = text_analysisV3.block_ranking(post_items=data, question=data['q'])

            # Step 3: Record new rank result
            result = db.insert_new_ranks({"question": data['q'], "ranks": ranks})
            print(result)
        response = {"ranks": ranks}

    except Exception as e:
        print(e.__context__)
        response = {"Error": e.__context__}
    return jsonify(response)


@text_analysis_api.route("/old_data_clean", methods=['POST'])
# content pre-process for single doc
def old_data_clean():
    data = request.get_json()
    try:
        print("Cleaning Document...")
        try:
            if data['version'] == 0:
                ta = text_analysisV0.TextAnalyze()
            elif data['version'] == 1:
                ta = text_analysisV1.TextAnalyze()
            else:
                ta = text_analysisV2.TextAnalyze()
        except KeyError:
            ta = text_analysisV2.TextAnalyze()
        tokens, doc = ta.content_pre_process(data['content'])
        response = {"token": tokens}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)


# API interface for old block analysis methods (Need to use v0~v2 data format)
@text_analysis_api.route("/old_block_analysis", methods=['POST'])
# block analysis
def old_block_analysis():
    data = request.get_json()
    try:
        print("Processing Block Analysis ...")
        try:
            print("Using Version " + str(data['version']) + " Method ...")
            if data['version'] == 0:
                ranks = text_analysisV0.block_ranking(stack_items=data['items'], qkey=data['qkey'])
            elif data['version'] == 1:
                ranks = text_analysisV1.block_ranking(stack_items=data['items'], question=data['q'])
            else:
                ranks = text_analysisV2.block_ranking(post_items=data['items'], question=data['q'])
        except KeyError:
            print("Using default version 3 ...")
            ranks = text_analysisV2.block_ranking(post_items=data['items'], question=data['q'])
        response = {"ranks": ranks}
    except Exception as e:
        response = {"error": e.__class__.__name__ + " : " + e.args[0]}
    return jsonify(response)


@text_analysis_api.route("/test", methods=['GET'])
def demo():
    with open("test/demo_example.json", 'r', encoding='utf-8') as file:
        test_data = json.load(file)
    return jsonify(test_data)





