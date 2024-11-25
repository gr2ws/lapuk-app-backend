from flask import Flask, request
from ultralytics import YOLO
import base64
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO("ai/garbage_detector_best_s.pt")


@app.route('/')
def connection_test():
    return "connected!"


@app.route('/detect', methods=['POST'])
def detect():
    received_image = cv2.imdecode(
        np.frombuffer(
            base64.b64decode(
                request.data
            ),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    result = model.predict(received_image, save=False)

    _, buffer = cv2.imencode('.jpg', result[0].plot())  # Extract the buffer directly
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__': app.run(debug=True, host='0.0.0.0')
