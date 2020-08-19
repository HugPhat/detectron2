import os
import urllib.request
import sys
from flask import Flask, request, redirect, jsonify, send_file
from werkzeug.utils import secure_filename

import numpy as np
import cv2
import urllib.request

import argparse

# import class inf
from maskrcnn_inf import maskrcnn_inf


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

###############################################

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store',
                    default=1,
                    dest='gpu',
                    help='Config CPU : 1 use | 0 not, default : 1')
parser.add_argument('--port', action='store',
                    default=8000,
                    dest='port',
                    help='Open port local host default : 8000')

results = parser.parse_args()
cpu = int(results.gpu)
port = int(results.port)

'''
CONFIG Model
    
    * maskrcnn_inf(model, cpu)
        * model:
            -------
            50FPN : lowest accuracy
            101FPN : good accuracy
            panoptic : good accuracy
            101X: good acc
            --------
        * cpu: when you switch to GPU -> cpu= False

'''
model = maskrcnn_inf(model='101X', cpu=True)

###############################################


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/test', methods=['POST'])
def run_inf_maskrcnn():
    if 'img' not in request.files:
        if type(request.json) == None or (not 'img' in list(request.json.keys())) or (type(request.json['img']) != str or request.json['img'] == ""):
            resp = jsonify({'message': 'img link is required'})
            resp.status_code = 400
            return resp
    try:
        img = request.files['img']
        img.save("./files/input.png")
        if not (img and allowed_file(img.filename)):
            resp = jsonify(
                {'message': 'Allowed file types are  jpg, jpeg, png'})
            resp.status_code = 400
            return resp

    except:
        img = request.json['img']
        urllib.request.urlretrieve(img, './files/input.png')

    finally:
        #imgfilename = secure_filename(img.filename)
        img = cv2.imread("./files/input.png")
        if type(img) != np.ndarray:
            resp = jsonify(
                {'message': 'Allowed file types are  jpg, jpeg'})
            resp.status_code = 400
            return resp
        output = model.predict(img=img, show=False)
        #
        #cv2.imwrite("./files/out.png", output)

        #return send_file("./files/out.png", mimetype='image/jpg')

        resp = jsonify(output)
        resp.status_code = 200
        return resp


if __name__ == '__main__':

    app.run(port=port, threaded=False, debug=False)
