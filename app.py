import cv2
import os
import insightface
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained detection model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load the swapper model
swapper = insightface.model_zoo.get_model(
    r'C:\Users\rshwa\Desktop\CodingProjects\Face_Swapper\inswapper_128.onnx',
    download=False, download_zip=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', swapped_image='', error_msg='')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', swapped_image='', error_msg='No file part.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', swapped_image='', error_msg='No selected file.')
    
    if file and allowed_file(file.filename):
        img1 = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check if the image contains exactly 2 faces
        faces = face_app.get(img1)
        if len(faces) != 2:
            error_msg = 'Image should contain exactly 2 faces.'
            return render_template('index.html', swapped_image='', error_msg=error_msg)
        
        face1, face2 = faces[0], faces[1]
        
        img1_ = img1.copy()
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img1_ = swapper.get(img1_, face2, face1, paste_back=True)
        
        # Convert the swapped image to base64 for displaying in the browser
        retval, buffer = cv2.imencode('.jpg', img1_)
        img_base64 = base64.b64encode(buffer).decode()
        
        return render_template('index.html', swapped_image=img_base64, error_msg='')
    
    error_msg = 'Unsupported file format. Please upload an image in PNG, JPG, JPEG, or GIF format.'
    return render_template('index.html', swapped_image='', error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)

