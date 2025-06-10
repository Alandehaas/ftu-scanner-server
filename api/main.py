from flask import Flask
from flask_cors import CORS
from api.routes import routes
import os

app = Flask(__name__)
app.register_blueprint(routes)
CORS(app)

# Ensure static upload directory exists
os.makedirs("static/uploads", exist_ok=True)

# This is only used if running with `python main.py`
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
