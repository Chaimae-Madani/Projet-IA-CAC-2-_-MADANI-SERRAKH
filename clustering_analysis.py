import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors

# Configuration générale
os.makedirs('images', exist_ok=True)
sns.set_theme(style="whitegrid")

print("=== DEBUT DE L'ANALYSE CLUSTERING ===")

# --- 1. K-Means ---
print("\n--- 1. K-MEANS (Iris) ---")
iris = datasets.load_iris()
X_iris = StandardScaler().fit_transform(iris.data)
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris)

inertias = []
silhouettes = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_iris)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_iris, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 10), inertias, marker='o', linewidth=2)
plt.title("K-Means : Méthode du Coude (Elbow) - Iris")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie intra-cluster (WCSS)")
plt.savefig('images/kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()

km_best = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km_best.fit_predict(X_iris)
print(f"Silhouette (k=3): {silhouette_score(X_iris, labels_km):.3f}")
print(f"Davies-Bouldin (k=3): {davies_bouldin_score(X_iris, labels_km):.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_iris_pca[:,0], y=X_iris_pca[:,1], hue=labels_km, palette='viridis', s=80)
plt.title("K-Means (k=3) : Projection PCA de l'Iris")
plt.savefig('images/kmeans_pca.png', dpi=150, bbox_inches='tight')
plt.close()


# --- 2. DBSCAN ---
print("\n--- 2. DBSCAN (Moons) ---")
X_moons, _ = datasets.make_moons(n_samples=300, noise=0.08, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_moons)
distances, _ = neighbors_fit.kneighbors(X_moons)
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(distances, linewidth=2)
plt.title("DBSCAN : K-Distance Graph (k=5)")
plt.ylabel("Distance au 5ème voisin le plus proche")
plt.xlabel("Points ordonnés")
plt.savefig('images/dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.close()

dbscan = DBSCAN(eps=0.25, min_samples=5)
labels_db = dbscan.fit_predict(X_moons)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)
print(f"Clusters: {n_clusters}, Bruit: {n_noise} ({n_noise/len(X_moons):.1%})")
mask = labels_db != -1
if n_clusters > 1:
    print(f"Silhouette (sans bruit): {silhouette_score(X_moons[mask], labels_db[mask]):.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_moons[:,0], y=X_moons[:,1], hue=labels_db, palette='Dark2', s=80)
plt.title("DBSCAN : Clustering sur Make_Moons")
plt.savefig('images/dbscan_moons.png', dpi=150, bbox_inches='tight')
plt.close()


# --- 3. CAH ---
print("\n--- 3. CAH (Wine) ---")
wine = datasets.load_wine()
X_wine = StandardScaler().fit_transform(wine.data)
pca_wine = PCA(n_components=2)
X_wine_pca = pca_wine.fit_transform(X_wine)

linkage_matrix = linkage(X_wine, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=4, leaf_rotation=90)
plt.title("CAH : Dendrogramme (Ward) - Wine dataset")
plt.ylabel("Distance de Ward")
plt.savefig('images/cah_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()

cah = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_cah = cah.fit_predict(X_wine)
print(f"Silhouette (k=3): {silhouette_score(X_wine, labels_cah):.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_wine_pca[:,0], y=X_wine_pca[:,1], hue=labels_cah, palette='Set1', s=80)
plt.title("CAH (k=3) : Projection PCA de Wine")
plt.savefig('images/cah_pca.png', dpi=150, bbox_inches='tight')
plt.close()


# --- 4. GMM ---
print("\n--- 4. GMM (Anisotropic Blobs) ---")
X_blobs, _ = datasets.make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=42)
# Transformation pour rendre les clusters anisotropes (étirés)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_blobs, transformation)
X_aniso = StandardScaler().fit_transform(X_aniso)

bics = []
aics = []
N_COMP = range(1, 7)
for n in N_COMP:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X_aniso)
    bics.append(gmm.bic(X_aniso))
    aics.append(gmm.aic(X_aniso))

plt.figure(figsize=(8, 5))
plt.plot(N_COMP, bics, label='BIC', marker='o', linewidth=2)
plt.plot(N_COMP, aics, label='AIC', marker='s', linewidth=2)
plt.title("GMM : Sélection du modèle (BIC/AIC)")
plt.xlabel("Nombre de composantes gaussiennes (k)")
plt.ylabel("Score")
plt.legend()
plt.savefig('images/gmm_bic.png', dpi=150, bbox_inches='tight')
plt.close()

gmm_best = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm_best.fit_predict(X_aniso)
print(f"Silhouette (k=3): {silhouette_score(X_aniso, labels_gmm):.3f}")

# Obtenir les probabilités d'appartenance pour illustrer le côté "soft clustering"
probs = gmm_best.predict_proba(X_aniso)
confidence = probs.max(axis=1)

# Le tracé met en évidence l'incertitude (les points moins sûrs sont plus petits/clairs)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_aniso[:,0], y=X_aniso[:,1], hue=labels_gmm, size=confidence**2, 
                sizes=(20, 100), palette='coolwarm', alpha=0.8)
plt.title("GMM (k=3) : Clustering Soft (La taille dépend de la confiance)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('images/gmm_pca.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n=== FIN DE L'ANALYSE ===")
