import os
# basic application framework
from flask import Flask
# blueprint register
from text_analysis_api import text_analysis_api
from data_manager_api import data_manager_api
from website import website

# CORS validation
from flask_cors import CORS


blueprint_prefix = [
    (text_analysis_api, "/api"),
    (data_manager_api, "/api"),
    (website, "/")
]

def register_blueprint(app):
    for blueprint, prefix in blueprint_prefix:
        app.register_blueprint(blueprint, url_prefix=prefix)
    return app


# set Flask application
app = Flask(__name__)
CORS(app)
register_blueprint(app)


if __name__ == "__main__":
    print("Welcome to MPAbotTools service system~~")
    app.run(host='0.0.0.0', port=os.environ.get("FLASK_SERVER_PORT", 8000), debug=True)
