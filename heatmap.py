import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from flask import Flask, request, send_file

app = Flask(__name__)

# csv file in 'data' folder
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'LAPUK_heatmapDataset.csv')

@app.route('/render-heatmap', methods=['GET'])
def generate_heatmap():
    try:
        df = pd.read_csv(CSV_FILE_PATH)

        columnLabels = ['Est. 2025 Population', '2020 Population', '2015 Population']
        
        # Convert columns to numeric, coercing errors 
        df[columnLabels] = df[columnLabels].apply(pd.to_numeric, errors='coerce')
        
        # Create the heatmap with only numeric data
        numeric_df = df[columnLabels]

        fig, ax = plt.subplots(figsize=(8, 4)) 
        cax = ax.imshow(numeric_df, cmap='coolwarm', aspect='auto')

        # Add color bar
        fig.colorbar(cax)

        # Set axis labels
        ax.set_xticks(np.arange(len(numeric_df.columns)))
        ax.set_xticklabels(numeric_df.columns, rotation=45, ha="right")
       
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df['Dumpsite Location'])

        # Value (cell text)
        for i in range(len(df)):
            for j in range(len(numeric_df.columns)):
                ax.text(j, i, f"{numeric_df.iloc[i, j]}", ha='center', va='center', color='black', fontsize=8)

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
    app.run(debug=True)
