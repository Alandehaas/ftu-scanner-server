from flask import Flask
from api.routes import routes
from flask_cors import CORS
import os

app = Flask(__name__)
app.register_blueprint(routes)
CORS(app)

os.makedirs("static/uploads", exist_ok=True)