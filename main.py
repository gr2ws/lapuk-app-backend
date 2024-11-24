from flask import Flask, request
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os
import shutil

app = Flask(__name__)

model = YOLO("ai/garbage_detector_best_s.pt")

def delete_predict():
    # Delete contents of the predict folder
    predict_folder = 'predict'
    for filename in os.listdir(predict_folder):
        file_path = os.path.join(predict_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def encode(image_path):
    img = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', img)  # Extract the buffer directly
    base64_encoded = base64.b64encode(buffer).decode('utf-8')
    return base64_encoded

@app.route('/')
def connection_test():
    return "connected!"


@app.route('/detect', methods=['POST'])
def detect():
    received_image = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(
                request.data.decode('utf-8')
            ),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    delete_predict()

    result = model.predict(received_image,
                           save=True,
                           save_dir='result',
                           project='./',
                           exist_ok=True)

    return encode('predict/image0.jpg')




if __name__ == '__main__': app.run(debug=True, host='0.0.0.0')
