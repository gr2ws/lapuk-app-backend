from flask import Flask, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("ai/garbage_detector_best_s.pt")

@app.route('/')
def connection_test():
    return "connected!"

if __name__ == '__main__': app.run(debug=True, host='0.0.0.0')