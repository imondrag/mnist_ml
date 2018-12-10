import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import predict_digit

PORT = 5050
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CLASSIFIER = predict_digit.DigitClassifier()
CLASSIFIER.read_training_checkpoint()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/classify", methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        img = request.files['file']

        if img.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if img and allowed_file(img.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename))
            img.save(filepath)
            guess = CLASSIFIER.predict_from_img(filepath)
            return 'Did you write a ' + str(guess) + '?'

    return '''
    <!doctype html>
    <title>Upload TEST</title>
    <h1>Upload to Classify</h1>
    <form method=post enctype=multipart/form-data>
        <p><input type=file name=file>
            <input type=submit value=Upload>
    </form>
    '''
