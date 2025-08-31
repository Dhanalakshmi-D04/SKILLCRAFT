import os
print("Current directory:", os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\dhana\Downloads\Mall_Customers.csv')

# Optional: Print column names to verify
print("Column names:", df.columns.tolist())

# 2. Preprocessing
# Encode gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# Select features (corrected column name)
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine optimal K using Elbow Method & Silhouette Score
wcss = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_pred = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, y_pred))

# Plot Elbow and Silhouette
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'go-')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Choose k (e.g., the one with highest silhouette score)
optimal_k = K_range[sil_scores.index(max(sil_scores))]
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

# 4. Fit K-Means with optimal_k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = components[:, 0], components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Cluster', palette='Set2',
    data=df, s=100, alpha=0.8
)
plt.title('Customer Segments via PCA Projection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()

# 6. Display cluster centroids in original feature space
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
print("Cluster centroids (original scale):\n", centroid_df)

# 7. Save the clustered data
df.to_csv('Mall_Customers_clustered.csv', index=False)
