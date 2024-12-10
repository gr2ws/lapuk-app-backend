import base64
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # to prevent frontend blocking GET requests
from matplotlib.colors import LinearSegmentedColormap  # color customization
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow all domains by default

model = YOLO("ai/gd_l_best.pt")

# CSV file in 'data' folder
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'LAPUK_heatmapDataset.csv')

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
    # noinspection PyTypeChecker
    plot_result = base64.b64encode(buffer).decode('utf-8')

    response = {'image': plot_result,
                'detections': [] if not list_detections else list_detections}

    return jsonify(response)

@app.route('/render-heatmap', methods=['GET'])
def generate_heatmap():
    """
    Endpoint to generate a heatmap from the CSV data.

    Returns:
        Response: A PNG image of the heatmap.
    """
    try:
        df = pd.read_csv(CSV_FILE_PATH)

        column_labels = ['Est. 2025 Population', '2020 Population', '2015 Population']

        # Convert columns to numeric, coercing errors
        df[column_labels] = df[column_labels].apply(pd.to_numeric, errors='coerce')

        # Create the heatmap with only numeric data
        numeric_df = df[column_labels]

        greens = LinearSegmentedColormap.from_list("greens", ["#E9F3E4", "#6B8E6B"])  # Light green to dark green

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(numeric_df, cmap=greens, aspect='auto')  # Apply the custom colormap

        fig.colorbar(cax)  # graph legend

        # Set axis labels
        ax.set_xticks(np.arange(len(numeric_df.columns)))
        ax.set_xticklabels(numeric_df.columns, rotation=30, ha="right")

        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df['Dumpsite Location'])

        # Value (cell text)
        for i in range(len(df)):
            for j in range(len(numeric_df.columns)):
                ax.text(j, i, f"{numeric_df.iloc[i, j]}", ha='center', va='center', color='black', fontsize=12)

        # Remove axis ticks
        ax.set_xticks(np.arange(len(numeric_df.columns)))
        ax.set_yticks(np.arange(len(df)))

        # Save the image to a file
        output_file = "heatmap.png"

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        return send_file(output_file, mimetype='image/png')

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')