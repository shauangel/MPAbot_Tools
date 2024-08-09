from flask import request, Blueprint, jsonify

# register api
data_manager_api = Blueprint("data_manager_api", __name__)


@data_manager_api.route("/insert_posts")
def insert_posts():
    return "success"