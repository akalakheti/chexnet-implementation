from flask import Flask, render_template, request,json, send_from_directory,send_file
import os
from werkzeug.utils import secure_filename
from backend import backends, cam
import cv2
import socket
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField, TextAreaField


app = Flask(__name__)

sckt = socket.gethostbyname(socket.gethostname())


UPLOAD_FOLDER = './images/upload'
HEATMAP_FOLDER = './images/heatmap'
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']



app.config['SECRET_KEY'] = 'abcdefgh'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER

@app.route('/')
def index():
    return render_template('index.html', sckt=sckt)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', sckt=sckt)

@app.route('/origin/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/heatmap/<filename>')
def send_image(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER']
                               , filename)

@app.route('/upload', methods = ['GET', 'POST'])
def main():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        print(filename)
        origin_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(origin_filename)
        file.save(origin_filename)
        lst = backends(origin_filename)
        img = cam(origin_filename)
        origin_filename2 = os.path.join(app.config['HEATMAP_FOLDER'], filename)
        cv2.imwrite(origin_filename2, img)
    
        return render_template('uploaded.html', image_name = filename, lst=lst, sckt=sckt)
    
        
    

        
        
    
        
        
app.run()
