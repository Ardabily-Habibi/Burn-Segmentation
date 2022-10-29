from importlib.metadata import metadata
import pickle
from fileinput import filename
from importlib.resources import path
from threading import local
from tkinter import Image
from tkinter.tix import IMAGE
from black import out
from flask import Flask, flash, request, redirect, url_for, render_template
from numpy import source
import urllib.request
import os
from werkzeug.utils import secure_filename

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from PIL import Image

 
app = Flask(__name__, template_folder='template')
 
UPLOAD_FOLDER = 'static/uploads/'
SEGMENTATION_FOLDER = 'static/segmentation'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTATION_FOLDER'] = SEGMENTATION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
# def model():
#     cfg = get_cfg()
#     cfg.MODEL.DEVICE = 'cpu'
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")  # path to the model we just trained
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = os.path.join("output/best_model.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

metadata = pickle.load(open('metadata.pkl', 'rb'))
 
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def get_predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        im = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        output = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], output))
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        im2 = Image.fromarray(out.get_image()[:, :, ::-1])
        cv2.imwrite(os.path.join(app.config['SEGMENTATION_FOLDER'], filename), out.get_image())
        # display_image = im2

        # flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_segment(filename):
    print("Gambar Asli")
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='segmentation/' + filename), code=301)
 
@app.route('/displya/<filename>')
def display_image(filename):
    print("Hasil Segmentasi")
    return redirect(url_for('static', filename='uploads/' + filename), code=302)

if __name__ == "__main__":
    print(metadata.as_dict())
    app.run()