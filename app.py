import flask
import io
import string
import time
import os
import cv2
import numpy as np

from PIL import Image
from flask import Flask, jsonify, request


from torch_utils import prediction, transform_image


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    # if 'file' not in request.files:
    # # if request.is_json == False:
    #         return "Please try again. The Image doesn't exist"
    
    # file = request.files.get('file')
    file = request.form.get('file')

    # if not file:
    #     return

    img_bytes = file
    img = transform_image(img_bytes)

    return jsonify(prediction=prediction(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Cataract Detection Using Deep Learning'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
