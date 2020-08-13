import os
import urllib.request
import sys
from flask import Flask, request, redirect, jsonify, send_file
from werkzeug.utils import secure_filename

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

model = maskrcnn_inf(cpu= cpu)

###############################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/test', methods=['POST'])
def run_inf_maskrcnn():
    if 'img' not in request.files:
        resp = jsonify({'message': 'img link is required'})
        resp.status_code = 400
        return resp
    try:
        img = request.files['img']
        if not (img and allowed_file(img.filename)):
            resp = jsonify(
                {'message': 'Allowed file types are  jpg, jpeg'})
            resp.status_code = 400
            return resp

    except:
        urllib.request.urlretrieve(img, './../files/input.png')
        img = cv2.imread("./../files/input.png")

    finally:
        #imgfilename = secure_filename(img.filename)
        output = model.predict(img=img, show=True)
        
        #
        cv2.imwrite(output.astype('uint8'), "./../files/out.png")

        return send_file("./../files/out.png")

if __name__ == '__main__':
    
    app.run(port=port, threaded=False, debug=False)
