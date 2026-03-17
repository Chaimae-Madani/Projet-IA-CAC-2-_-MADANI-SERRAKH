import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors

# ==========================================
# CONFIGURATION GLOBALE
# ==========================================
os.makedirs('images', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

print("=== DEBUT DE L'ANALYSE EXPERTE DE CLUSTERING ===")

# ==========================================
# 1. K-MEANS (Jeu de données : Iris)
# ==========================================
print("\n--- 1. K-MEANS (Iris Dataset) ---")
iris = datasets.load_iris()
X_iris = iris.data
# BONNE PRATIQUE : Normalisation
X_iris_s = StandardScaler().fit_transform(X_iris)

# PCA pour la visualisation
pca_iris = PCA(n_components=2)
X_iris_2d = pca_iris.fit_transform(X_iris_s)

# A. Méthode du Coude et Silhouette pour choisir K
inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_iris_s)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_iris_s, labels))

# Graphique : Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o', color='crimson', linewidth=2.5, markersize=8)
plt.title("K-Means : Méthode du Coude (Elbow) sur Iris", fontweight='bold')
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie intra-cluster (WCSS)")
plt.savefig('images/kmeans_elbow.png', dpi=300, bbox_inches='tight')
plt.close()

# B. Modèle final avec K=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = km_final.fit_predict(X_iris_s)

print(f"Silhouette (K=3): {silhouette_score(X_iris_s, labels_km):.3f}")
print(f"Davies-Bouldin (K=3): {davies_bouldin_score(X_iris_s, labels_km):.3f}")
print(f"Calinski-Harabasz (K=3): {calinski_harabasz_score(X_iris_s, labels_km):.1f}")

# Graphique : Projection des clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_iris_2d[:, 0], y=X_iris_2d[:, 1], hue=labels_km, palette="Set1", s=100, edgecolor='k')
# Affichage des centroïdes
centers_2d = pca_iris.transform(km_final.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=250, marker='X', label='Centroïdes')
plt.title("K-Means (K=3) : Projection PCA des Clusters (Iris)", fontweight='bold')
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.legend()
plt.savefig('images/kmeans_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. DBSCAN (Jeu de données : Make Moons)
# ==========================================
print("\n--- 2. DBSCAN (Make Moons Dataset) ---")
# Dataset non-convexe parfait pour DBSCAN
X_moons, _ = datasets.make_moons(n_samples=400, noise=0.08, random_state=42)
X_moons_s = StandardScaler().fit_transform(X_moons)

# A. K-Distance Graph pour trouver le bon epsilon (eps)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_moons_s)
distances, indices = neighbors_fit.kneighbors(X_moons_s)
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(distances, color='darkorange', linewidth=2.5)
plt.axhline(y=0.25, color='r', linestyle='--', label='Seuil optimal eps ≈ 0.25')
plt.title("DBSCAN : K-Distance Graph (K=5 voisins)", fontweight='bold')
plt.ylabel("Distance au 5ème voisin le plus proche")
plt.xlabel("Points triés par distance")
plt.legend()
plt.savefig('images/dbscan_kdistance.png', dpi=300, bbox_inches='tight')
plt.close()

# B. Modèle final
dbscan = DBSCAN(eps=0.25, min_samples=5)
labels_db = dbscan.fit_predict(X_moons_s)

n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)
print(f"Clusters trouvés : {n_clusters_db}")
print(f"Points de bruit  : {n_noise} sur {len(X_moons_s)} ({n_noise/len(X_moons_s):.1%})")

# Evaluation sans le bruit
mask = labels_db != -1
if n_clusters_db > 1:
    print(f"Silhouette (sans bruit): {silhouette_score(X_moons_s[mask], labels_db[mask]):.3f}")

# Graphique : Visualisation des clusters (+ bruit en noir)
plt.figure(figsize=(8, 5))
palette_db = sns.color_palette("husl", n_clusters_db)
# Ajout de la couleur noire pour le bruit
palette_db_with_noise = ['black'] + list(palette_db) if -1 in labels_db else palette_db
sns.scatterplot(x=X_moons_s[:,0], y=X_moons_s[:,1], hue=labels_db, palette=palette_db_with_noise, s=80, edgecolor='k')
plt.title("DBSCAN : Clustering de formes non-convexes (Moons)", fontweight='bold')
plt.xlabel("Feature 1 (Standardisée)")
plt.ylabel("Feature 2 (Standardisée)")
plt.legend(title='Cluster / Bruit (-1)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('images/dbscan_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. CAH - CLASSIFICATION ASCENDANTE HIERARCHIQUE (Jeu de données : Wine)
# ==========================================
print("\n--- 3. CAH (Wine Dataset) ---")
wine = datasets.load_wine()
X_wine = wine.data
X_wine_s = StandardScaler().fit_transform(X_wine)
pca_wine = PCA(n_components=2)
X_wine_2d = pca_wine.fit_transform(X_wine_s)

# A. Dendrogramme (Méthode de Ward)
linkage_matrix = linkage(X_wine_s, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5, leaf_rotation=90, color_threshold=15)
plt.axhline(y=15, color='r', linestyle='--', label='Coupe pour K=3')
plt.title("CAH : Dendrogramme avec liaison de Ward (Wine)", fontweight='bold')
plt.ylabel("Distance (Ward)")
plt.xlabel("Index des points ou (taille des sous-clusters)")
plt.legend()
plt.savefig('images/cah_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()

# B. Modèle final avec K=3
cah = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_cah = cah.fit_predict(X_wine_s)

print(f"Silhouette (K=3): {silhouette_score(X_wine_s, labels_cah):.3f}")
print(f"Davies-Bouldin (K=3): {davies_bouldin_score(X_wine_s, labels_cah):.3f}")

# Graphique : Projection PCA
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_wine_2d[:,0], y=X_wine_2d[:,1], hue=labels_cah, palette="magma", s=100, edgecolor='k')
plt.title("CAH (K=3) : Projection PCA des Clusters (Wine)", fontweight='bold')
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.legend(title='Cluster')
plt.savefig('images/cah_clusters_pca.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. GAUSSIAN MIXTURE MODELS - GMM (Jeu de données : Blobs anisotropes)
# ==========================================
print("\n--- 4. GMM (Anisotropic Blobs Dataset) ---")
# Génération de clusters elliptiques (où K-Means échouerait)
X_blobs, _ = datasets.make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=42)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X_blobs, transformation)
X_aniso_s = StandardScaler().fit_transform(X_aniso)

# A. Sélection du nombre de composantes avec BIC/AIC
bics = []
aics = []
N_COMP = range(1, 7)

for n in N_COMP:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X_aniso_s)
    bics.append(gmm.bic(X_aniso_s))
    aics.append(gmm.aic(X_aniso_s))

plt.figure(figsize=(8, 5))
plt.plot(N_COMP, bics, label='BIC', marker='o', color='navy', linewidth=2.5)
plt.plot(N_COMP, aics, label='AIC', marker='s', color='teal', linewidth=2.5)
plt.title("GMM : Critères d'Information BIC et AIC", fontweight='bold')
plt.xlabel("Nombre de composantes gaussiennes (K)")
plt.ylabel("Score d'Information (Le plus bas est le meilleur)")
plt.legend()
plt.savefig('images/gmm_bic_aic.png', dpi=300, bbox_inches='tight')
plt.close()

# B. Modèle final avec K=3
gmm_final = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm_final.fit_predict(X_aniso_s)
# Probabilités pour la notion de "Soft Clustering"
probs = gmm_final.predict_proba(X_aniso_s)
confidences = probs.max(axis=1)

print(f"Silhouette (K=3): {silhouette_score(X_aniso_s, labels_gmm):.3f}")

# Graphique : Soft Clustering où la taille (ou l'alpha) dépend de la certitude
plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_aniso_s[:, 0], X_aniso_s[:, 1], 
                      c=labels_gmm, cmap='viridis', 
                      s=30 + 150 * confidences**4, # Amplification visuelle de la certitude
                      alpha=0.7, edgecolor='k')
plt.title("GMM (K=3) : Clustering Elliptique et Confiance", fontweight='bold')
plt.xlabel("Feature 1 (Standardisée)")
plt.ylabel("Feature 2 (Standardisée)")
# Ajout d'une colorbar générique pour représenter les classes
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)
plt.figtext(0.5, 0.01, "NB: Les points plus petits indiquent une faible confiance d'appartenance", 
            wrap=True, horizontalalignment='center', fontsize=10, style='italic')
plt.savefig('images/gmm_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== GENERATION DES GRAPHIQUES TERMINEE (DOSSIER 'images/') ===")
