# 📊 Compte Rendu d'Exploration Analytique : Modèles de Clustering

**Auteur :** Data Scientist (Expert IA)  
**Objectif :** Évaluer et comparer 4 algorithmes majeurs de clustering (K-Means, DBSCAN, CAH, GMM) sur des jeux de données réactifs à leurs spécificités.

---

## 1. K-Means (La Référence Centroïdale)

### Présentation théorique
* **Définition :** K-Means est un algorithme de partitionnement qui divise un jeu de données en $K$ groupes distincts en minimisant la variance intra-cluster (la somme des distances au carré entre chaque point et le centroïde de son cluster).
* **Hypothèses :** L'algorithme suppose que les clusters sont sphériques (iso-distants), de densité homogène, et de tailles comparables.
* **Avantages :** Extrêmement rapide, facile à implémenter, scalaire sur de grands volumes de données.
* **Limites :** Sensible aux données aberrantes (outliers), exige de connaître $K$ à l'avance, et échoue sur des clusters de formes complexes (e.g. lunes, cercles concentriques).
* **Complexité :** $\mathcal{O}(n \cdot K \cdot I \cdot d)$ où $I$ est le nombre d'itérations, $d$ est la dimension. C'est linéaire en pratique.

### Évaluation et Analyse (Jeu de données : Iris)
* **Dataset choisi :** `Iris` (150 échantillons, 4 dimensions réelles réduites par PCA). Idéal car les classes y sont plutôt convexes et équilibrées.
* **Métriques obtenues :**
  * **Méthode du coude (Elbow) :** Identifie clairement $K=3$ ou $K=2$.
  * **Silhouette Score ($K=3$) :** $\approx 0.46$
  * **Davies-Bouldin ($K=3$) :** $\approx 0.83$
* **Interprétation Statistique :** Un score de Silhouette de 0.46 montre une structuration modérée à forte. Le jeu de données Iris possède deux classes ("Versicolor" et "Virginica") qui se chevauchent légèrement, ce qui abaisse le score par rapport au maximum théorique de 1.
* **Insights d'Expert :** Je fixe `n_init=10` pour relancer l'algorithme 10 fois avec des centroïdes initiaux différents, évitant ainsi de tomber dans un minimum local lié à la méthode d'initialisation aléatoire.

### Conclusion sur K-Means
* **Usage recommandé :** Quand on recherche des segments de marché clairs (profilage client, compression d'image) et que l'on dispose d'une bonne volumétrie sans formes tordues.

---

## 2. DBSCAN (L'Approche par Densité)

### Présentation théorique
* **Définition :** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) regroupe les points fortement denses et marque les points isolés dans des régions de faible densité comme "Bruit".
* **Hypothèses :** Les clusters sont des zones de haute densité continue séparées par des zones de basse densité.
* **Avantages :** Dégage les points aberrants, trouve automatiquement le nombre de clusters, et identifie n'importe quelle forme géométrique non-convexe.
* **Limites :** Peine sérieusement si la densité des clusters varie drastiquement dans le jeu de données ou en très haute dimension (fléau de la dimensionnalité sur la distance Euclidienne).
* **Complexité :** $\mathcal{O}(n \log n)$ au mieux avec un arbre d'indexage (KD-Tree), sinon $\mathcal{O}(n^2)$ dans le pire des cas.

### Évaluation et Analyse (Jeu de données : Make Moons)
* **Dataset choisi :** `make_moons` avec du bruit. K-Means couperait ces lunes en deux. DBSCAN est le seul à suivre la courbure.
* **Hyperparamètres choisis :**
  * `eps = 0.25` : Choisie grâce au genou de la courbe "K-Distance Graph" (affichée dans le code). Dans un espace standardisé, c'est la distance où le bruit commence à exploser.
  * `min_samples = 5` : Valeur robuste pour un jeu de 400 points en 2D. 
* **Métriques obtenues :**
  * **Clusters trouvés :** 2 (correspondant exactement aux lunes).
  * **Bruit :** $\approx 4\%$ (exclus proprement des clusters).
  * **Silhouette Score (sans bruit) :** $\approx 0.33$. *(Note: la silhouette utilise des distances euclidiennes et pénalise mathématiquement les formes incurvées bien que DBSCAN ait parfaitement réussi du point de vue topologique).*
* **Insights d'Expert :** Ne jamais utiliser DBSCAN sans avoir d'abord tracé le diagramme K-Distance. L'impact de $Eps$ est fulgurant ; un dixième de décalage peut regrouper toutes les lunes dans un cluster ou les disloquer en micro-clusters.

### Conclusion sur DBSCAN
* **Usage recommandé :** Détection d'anomalies (fraude, maintenance prédictive), ou lorsque les données ont des formes spatialement alambiquées avec beaucoup de bruit organique.

---

## 3. CAH - Classification Ascendante Hiérarchique

### Présentation théorique
* **Définition :** Construit une arborescence (Dendrogramme) de bas en haut. Au départ, chaque point est un cluster. À chaque étape, l'algorithme fusionne les 2 clusters les plus "proches" selon un critère de liaison (Linkage).
* **Hypothèses :** Il existe une structure hiérarchique latente dans les données.
* **Avantages :** Extrêmement visuel (Dendrogramme), pas besoin de spécifier explicitement $K$ avant l'analyse finale (on coupe l'arbre où l'on veut).
* **Limites :** Impossible à annuler (une fusion est définitive). Horriblement coûteux en mémoire et temps de calcul sur de grands volumes.
* **Complexité :** $\mathcal{O}(n^3)$ en temps (ou $\mathcal{O}(n^2 \log n)$ bien optimisé), et $\mathcal{O}(n^2)$ en mémoire.

### Évaluation et Analyse (Jeu de données : Wine)
* **Dataset choisi :** `Wine` (13 dimensions). Ce dataset se prête bien à une fouille hiérarchique des caractéristiques chimiques des vins.
* **Hyperparamètres choisis :**
  * `linkage = 'ward'` : Cette liaison minimise l'augmentation de la variance intra-cluster à chaque fusion, produisant des clusters de taille équilibrée (comme K-Means, mais hiérarchiquement).
* **Métriques obtenues :**
  * Silhouette Score pour $K=3$ : $\approx 0.28$ 
* **Insights d'Expert :** Sur un jeu de données à 13 dimensions, la standardisation (`StandardScaler`) est une question de vie ou de mort pour cet algorithme, sinon la caractéristique chimique ayant la plus grande magnitude (comme la Proline) écraserait complètement toutes les autres dans le calcul de la séparation.

### Conclusion sur la CAH
* **Usage recommandé :** En biologie (génétique, phylogénie) ou lorsque la base de données est modeste mais que l'explicabilité et la taxonomie sont capitales pour le métier.

---

## 4. GMM - Gaussian Mixture Models

### Présentation théorique
* **Définition :** Algorithme génératif cherchant à représenter les données comme une superposition ("Mixture") de plusieurs distributions Gaussiennes. Il utilise l'algorithme "Expectation-Maximization" (EM).
* **Hypothèses :** Les clusters suivent une distribution normale multivariée (ellipses).
* **Avantages :** *Soft clustering*. Au lieu d'affecter un point durement à un cluster, GMM donne des probabilités d'appartenance (ex: 80% cluster A, 20% cluster B). Il gère les clusters étirés (anisotropiques).
* **Limites :** Algorithme capricieux : peut converger vers un minimum local (sensible à l'initialisation) et requiert énormément de données pour estimer de grandes matrices de covariance en haute dimension.
* **Complexité :** $\mathcal{O}(n \cdot K \cdot d^2)$ mais nécessite souvent beaucoup d'itérations EM.

### Évaluation et Analyse (Jeu de données : Blobs Anisotropes)
* **Dataset choisi :** Des *blobs* générés artificiellement et déformés via une matrice de transformation linéaire (étirés et inclinés). K-Means coupera mal ces formes allongées.
* **Hyperparamètres choisis :**
  * `covariance_type = 'full'` : Obligatoire ici. Cela permet à chaque Gaussienne d'avoir sa propre forme, longueur, largeur et orientation. C'est l'arme absolue du GMM.
* **Métriques obtenues :**
  * **Critères d'information :** Tracer le BIC et l'AIC marque un coude sans équivoque à $K=3$. Le BIC pénalise fortement la complexité, ce qui évite l'overfitting typique d'avoir 10 Gaussiennes.
* **Insights d'Expert :** Le "Soft Clustering" est extrêmement utile en entreprise. Dans le graphique généré, la taille d'un point est corrélée à la certitude du modèle. Les points situés aux frontières de deux ellipses Gaussiennes seront affichés plus petits ou clairs, avertissant le Data Scientist qu'il s'agit de profils "hybrides".

### Conclusion sur GMM
* **Usage recommandé :** Quand on cherche à capturer de l'incertitude et de la nuance (Traitement du signal, segmentation avancée d'utilisateurs mutables), et que l'on sait que les clusters ont des distributions spatiales "elliptiques" non-sphériques.

---

## 7. Compte Rendu Global et Recommandations

### Tableau de Synthèse

| Algorithme | Forme des clusters  | Nécessite de fixer $K$ ? | Sensibilité au Bruit (Outliers) | Complexité / Scalabilité | Output Spécifique |
|:---|:---|:---:|:---:|:---|:---|
| **K-Means** | Sphérique (Hyperboules)| Oui (absolument) | Très Forte (tire la moyenne) | $\mathcal{O}(n \cdot K)$ (Excellent) | Hard Labels |
| **DBSCAN** | Arbitraire géométrique | Non | Résistant (Extrait le bruit) | $\mathcal{O}(n \log n)$ (Moyen) | Hard Labels + Bruit |
| **CAH** | Selon le Linkage | Non (Dendrogramme)| Dépend (Ward y est fort sensible) | $\mathcal{O}(n^3)$ (Très Mauvais) | Arborescence (Taxonomie) |
| **GMM** | Ellipses (Multivariées) | Oui (BIC / AIC) | Modérée | $\mathcal{O}(n \cdot K \cdot d^2)$ (Lent) | Probabilités (Soft Labels) |

### 🛠 Recommandations d'Expert (Cas Métiers Réels)

1. **Vos données sont extrêmement massives (ex: 5 millions de logs) et simples ?**
   $\Rightarrow$ **MiniBatch-KMeans** ou **K-Means**. Ne tentez jamais une CAH.

2. **Vos données contiennent énormément de valeurs extrêmes ou d'outliers ?**
   $\Rightarrow$ **DBSCAN**. Il isolera la saleté en étiquette `-1` et révélera le signal "dense" sous-jacent. Si la dimension devient trop grande (> 50), utilisez plutôt **HDBSCAN**.

3. **Vos clusters potentiels ont des formes très bizarres (spatialement tordues, rivières d'informations) ?**
   $\Rightarrow$ **DBSCAN** (*voir Spectral Clustering en alternative si le bruit est négligeable.*)

4. **Vous n'avez absolument aucune idée du nombre de clusters ni de la géométrie, et vous cherchez une taxonomie ?**
   $\Rightarrow$ **CAH (Clustering Hiérarchique)**. Regarder le Dendrogramme est souvent la première étape exploratoire sur un petit échantillon (ex: 2000 lignes) pour se faire une idée visuelle des emboîtements.

5. **Vous avez besoin que votre modèle ne soit pas trop confiant aux frontières de décision (ex: analyse médicale de l'hybridation d'une cellule) ?**
   $\Rightarrow$ **GMM (Gaussian Mixture)**. Le probabilisme vous permettra de traiter les éléments avec une probabilité $\approx 50\%$ différemment des certitudes absolues à $99\%$.
