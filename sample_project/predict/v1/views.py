from flask import request, jsonify
from flask_restx import Resource
import numpy as np
import base64
import json
import cv2
import imutils
from imutils.contours import sort_contours
import requests

from sample_project.predict import ns


@ns.route('/predicts', methods=['POST'])
class Predict(Resource):
    def post(self):
        # Decoding and pre-processing base64 image
        IMAGE_SIZE = (700, 700)
        img_array = np.fromstring(base64.b64decode(request.form['b64']), np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(img_array, IMAGE_SIZE)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 150)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sort_contours(cnts, method="left-to-right")[0]
        chars = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            if (w >= 20 and w <= 150) and (h >= 60 and h <= 150):

                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape

                if tW > tH:
                    thresh = imutils.resize(thresh, width=28)

                else:
                    thresh = imutils.resize(thresh, height=28)

                (tH, tW) = thresh.shape
                dX = int(max(0, 28 - tW) / 2.0)
                dY = int(max(0, 28 - tH) / 2.0)

                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0))
                padded = cv2.resize(padded, (28, 28))

                padded = padded.astype("float32") / 255.0
                padded = np.expand_dims(padded, axis=-1)

                chars.append((padded, (x, y, w, h)))

        chars = np.array([c[0] for c in chars], dtype="float32")

        # Creating body for TensorFlow serving request
        data = json.dumps({"signature_name": "serving_default", "instances": chars.tolist()})
        print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

        headers = {"content-type": "application/json"}

        # Making POST request
        r = requests.post('http://number-model-container.herokuapp.com/v1/models/save_model:predict', data=data, headers=headers)

        # Decoding results from TensorFlow Serving server
        preds = json.loads(r.content.decode('utf-8'))

        labelNames = "0123456789"
        labelNames = [l for l in labelNames]

        data = []

        for pred in preds['predictions']:
            i = np.argmax(pred)
            prob = pred[i]
            label = labelNames[i]
            item = {"character": label, "score": prob * 100}
            data.append(item)

            print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        
        return jsonify(data)