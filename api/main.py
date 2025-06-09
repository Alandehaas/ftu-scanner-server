from flask import Flask
from api.routes import routes
import os
from flask_cors import CORS

app = Flask(__name__)
app.register_blueprint(routes)
CORS(app)

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

