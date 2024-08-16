from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

app = Flask(__name__)

# Load the K-means model
model_path = 'kmeans_customer_segmentation.pkl'
kmeans = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        # Load the dataset
        data = pd.read_csv(file)
        
        # Preprocess the dataset
        features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Predict clusters
        data['Cluster'] = kmeans.predict(scaled_features)
        
        # Visualize the clusters
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=data['Annual Income (k$)'], y=data['Spending Score (1-100)'], hue=data['Cluster'], palette='viridis')
        plt.title('Customer Segments')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        
        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        return send_file(img, mimetype='image/png')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
