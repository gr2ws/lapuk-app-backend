import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("ai/garbage_detector_best_s.pt")


@app.route('/')
def connection_test():
    """
    Endpoint to test the connection to the server.

    Returns:
        str: A simple message indicating the connection status.
    """
    return "connected!"


@app.route('/detect', methods=['POST'])
def detect():
    """
    Endpoint to perform object detection on an uploaded image.

    The image is expected to be sent as a base64-encoded string in the request body.
    The function decodes the image, runs the YOLO model to detect objects, and returns
    the detection results along with the image with plotted detections.

    Returns:
        Response: A JSON response containing the base64-encoded image with plotted detections
                  and a list of detected objects with their confidence scores.
    """
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
    detections = result[0].to_df().to_dict(orient='records')

    list_detections = []

    for detection in detections:
        list_detections.append((detection['name'], detection['confidence']))

    _, buffer = cv2.imencode('.jpg', result[0].plot())  # Extract the buffer directly
    plot_result = base64.b64encode(buffer).decode('utf-8')

    response = {'image': plot_result,
                'detections': [] if not list_detections else list_detections}

    return jsonify(response)


if __name__ == '__main__': app.run(debug=True, host='0.0.0.0')
