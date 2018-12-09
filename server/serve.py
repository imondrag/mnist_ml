from flask import Flask
app = Flask(__name__)

PORT = 5050

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/classify", methods=['POST'])
def classify():
    return "Hello World!"
