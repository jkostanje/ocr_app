from flask import request
from flask_restx import Resource
import numpy as np
import base64
import json
import cv2
import requests

from sample_project.predict import ns


@ns.route('/predicts', methods=['POST'])
class Predict(Resource):
    def post(self):
        # Decoding and pre-processing base64 image
        IMAGE_SIZE = (200, 200)
        img_array = np.fromstring(base64.b64decode(request.form['b64']), np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, IMAGE_SIZE, 1)
        new_array = new_array / 255

        # Creating body for TensorFlow serving request
        data = json.dumps({"signature_name": "serving_default", "instances": new_array.reshape(-1, 200, 200, 1).tolist()})
        print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

        headers = {"content-type": "application/json"}

        # Making POST request
        r = requests.post('https://wine-model-tfs.herokuapp.com/v1/models/wine_model:predict', data=data, headers=headers)

        # Decoding results from TensorFlow Serving server
        return json.loads(r.content.decode('utf-8'))
