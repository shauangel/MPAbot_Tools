import time

from flask import Blueprint, render_template, jsonify
import json

# register api
website = Blueprint("website", __name__)


@website.route("/hello", methods=['GET'])
def hello():
    return render_template('index.html')


@website.route("/sleep", methods=['GET'])
def test_waiting():
    time.sleep(30)
    return jsonify({"text": "finished waiting"})