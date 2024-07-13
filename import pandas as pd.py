import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# Display the first six rows
print(customer_data.head())

# Summary statistics
print(customer_data.describe())

# Standard deviation
print("Standard Deviation of Age:", np.std(customer_data['Age']))
print("Standard Deviation of Annual Income:", np.std(customer_data['Annual Income (k$)']))
print("Standard Deviation of Spending Score:", np.std(customer_data['Spending Score (1-100)']))

# Gender Visualization
gender_count = customer_data['Gender'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=gender_count.index, y=gender_count.values, hue=gender_count.index, palette='viridis', legend=False)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Gender Distribution')
plt.show()

# Histogram of Age
plt.figure(figsize=(12, 6))
plt.hist(customer_data['Age'], bins=30, color='blue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot of Age
plt.figure(figsize=(12, 6))
sns.boxplot(customer_data['Age'], color='pink')
plt.title('Boxplot of Age')
plt.show()

# Histogram of Annual Income
plt.figure(figsize=(12, 6))
plt.hist(customer_data['Annual Income (k$)'], bins=30, color='#660033', edgecolor='black')
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()

# Density Plot of Annual Income
plt.figure(figsize=(12, 6))
sns.kdeplot(customer_data['Annual Income (k$)'], fill=True, color='orange')
plt.title('Density Plot of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.show()

# Boxplot of Spending Score
plt.figure(figsize=(12, 6))
sns.boxplot(customer_data['Spending Score (1-100)'], color='#990000')
plt.title('Boxplot of Spending Score')
plt.show()

# Histogram of Spending Score
plt.figure(figsize=(12, 6))
plt.hist(customer_data['Spending Score (1-100)'], bins=30, color='#6600cc', edgecolor='black')
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.show()

# Standardizing the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Finding the optimal number of clusters using the Elbow Method
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 11))
visualizer.fit(customer_data_scaled)
visualizer.show()

# Applying K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(customer_data_scaled)

# Adding the cluster column to the original dataset
customer_data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=customer_data, palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
