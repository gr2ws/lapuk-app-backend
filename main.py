from flask import Flask, request
import base64
import cv2
import numpy as np
from ultralytics import YOLO

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
                request.data.decode('utf-8')
            ),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    result = model.predict(received_image, save=True)

    print(result)


if __name__ == '__main__': app.run(debug=True, host='0.0.0.0')
