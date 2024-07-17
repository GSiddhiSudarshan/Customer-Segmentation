# Import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

# Load customers data
customersdata = pd.read_csv("customers-data.csv")

# Define K-means model
kmeans_model = KMeans(init='k-means++', max_iter=400, random_state=42)

# Train the model
kmeans_model.fit(customersdata[['products_purchased', 'complains', 'money_spent']])

# Create the K means model for different values of K
def try_different_clusters(K, data):
    cluster_values = list(range(1, K+1))
    inertias = []

    for c in cluster_values:
        model = KMeans(n_clusters=c, init='k-means++', max_iter=400, random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias

# Find output for k values between 1 to 12
outputs = try_different_clusters(12, customersdata[['products_purchased', 'complains', 'money_spent']])
distances = pd.DataFrame({"clusters": list(range(1, 13)), "sum of squared distances": outputs})

# Finding optimal number of clusters k using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(distances["clusters"], distances["sum of squared distances"], marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.title("Finding optimal number of clusters using elbow method")
plt.xticks(list(range(1, 13)))
plt.grid(True)
plt.show()

# Re-Train K means model with k=5
kmeans_model_new = KMeans(n_clusters=5, init='k-means++', max_iter=400, random_state=42)
customersdata['clusters'] = kmeans_model_new.fit_predict(customersdata[['products_purchased', 'complains', 'money_spent']])

# Visualize clusters using Plotly
figure = px.scatter_3d(customersdata,
                    color='clusters',
                    x="products_purchased",
                    y="complains",
                    z="money_spent",
                    category_orders={"clusters": ["0", "1", "2", "3", "4"]}
                    )
figure.update_layout()
figure.show()

