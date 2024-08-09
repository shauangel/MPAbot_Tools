#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

from pymongo import MongoClient
from bson import json_util


test_client = MongoClient('mongodb://localhost:50000/')
DB = test_client["MPAbotTools"]
RESULT_COLLECTION = DB["Results"]
# RESULT_COLLECTION = DB["test-Results"]
QUESTION_COLLECTION = DB["Questions"]


# Query Previous Ranks
def query_previous_ranks(question):
    try:
        query = {"question": question}
        result = [r for r in RESULT_COLLECTION.find(query)]
        if result is not None:
            response = json.loads(json_util.dumps(result))
        else:
            response = {}
    except Exception as e:
        print(e.__context__)
        response = {"error": e.__context__}
    return response


# Insert New Ranks
def insert_new_ranks(ranks):
    try:
        inserted_ids = RESULT_COLLECTION.insert_one(ranks)
    except Exception as e:
        print(e.__context__)
    return str(inserted_ids)


# Query Related Links
# Future work: question similarity compare
def query_question(question):
    try:
        query = {"question": question}
        result = [r for r in QUESTION_COLLECTION.find(query)]
        if result is not None:
            response = json.loads(json_util.dumps(result))
        else:
            response = {}
    except Exception as e:
        print(e.__context__)
        response = {"error": e.__context__}
    return response


# Insert one question and related Links
def insert_one_question(item):
    try:
        # Step 1: Search for related questions (future work)
        """..."""
        # Step 2: Insert question
        item['similar_q'] = []
        inserted_id = QUESTION_COLLECTION.insert_one(item)
    except Exception as e:
        print(e.__context__)
    return str(inserted_id)


