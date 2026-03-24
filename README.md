# 🔍 Prédiction du Turnover des Auditeurs Internes

> **Projet Machine Learning — CAC 2 · S8 · ENCG Settat**  
> Application du Machine Learning à la Gestion des Ressources Humaines

---

## 👥 Auteurs

| Nom | Filière |
|-----|---------|
| **Madani Chaimae** | CAC 2 — S8 |
| **Serrakh Nohaila** | CAC 2 — S8 |

**Encadrant :** Pr. Abderrahim Larhlimi  
**Institution :** École Nationale de Commerce et de Gestion de Settat — Université Hassan 1er  
**Période :** Mars 2025

---

## 📌 Description du projet

Ce projet prédit le **départ d'auditeurs internes dans les 12 prochains mois** à l'aide d'algorithmes de Machine Learning supervisé. Il s'inscrit dans une approche de **Predictive HR Analytics** visant à aider les cabinets d'audit marocains à anticiper et réduire le turnover de leurs équipes.

---

## 📊 Données

| Paramètre | Valeur |
|-----------|--------|
| **Taille de l'échantillon** | 500 auditeurs |
| **Nombre de variables** | 34 (dont variable cible) |
| **Variable cible** | `Depart_12mois` (Oui/Non) |
| **Taux de turnover** | 38 % (190 départs / 310 restants) |
| **Source** | Données simulées à des fins pédagogiques |

### Blocs thématiques couverts
- 👤 Démographie (âge, genre, ville, niveau d'études)
- 💼 Profil professionnel (poste, ancienneté, type de cabinet)
- 😊 Satisfaction au travail (globale, manager, rémunération, stress…)
- 💰 Rémunération (salaire, bonus, augmentation)
- 📈 Développement de carrière (formations, promotions, mentoring)
- 🔍 Comportements observables (entretiens extérieurs, absences, mobilité)

---

## ⚙️ Méthodologie

### Prétraitement
- **Encodage** des variables catégorielles (One-Hot Encoding + Label Encoding)
- **Normalisation** (StandardScaler) pour les modèles sensibles à l'échelle
- **Découpage** : 80 % entraînement / 20 % test (stratifié)
- **Gestion du déséquilibre** : `class_weight='balanced'`

### Algorithmes testés

| Modèle | Type |
|--------|------|
| Régression Logistique | Linéaire — baseline |
| Decision Tree | Arbre de décision |
| **Random Forest** ✅ | Ensemble — **modèle retenu** |
| XGBoost | Boosting |
| SVM (noyau RBF) | Hyperplan à marge maximale |
| KNN (k=5) | Voisinage |

---

## 📈 Résultats

### Comparaison des performances (jeu de test, N=100)

| Modèle | Accuracy | AUC-ROC | F1-Score (Oui) |
|--------|----------|---------|----------------|
| **Random Forest** | **83 %** | **0,91** | **0,78** |
| XGBoost | 82 % | 0,90 | 0,77 |
| SVM (RBF) | 76 % | 0,83 | 0,69 |
| Régression Logistique | 74 % | 0,81 | 0,67 |
| Decision Tree | 73 % | 0,72 | 0,65 |
| KNN (k=5) | 70 % | 0,76 | 0,62 |

### Modèle sélectionné : Random Forest
- **Accuracy :** 83 % | **AUC-ROC :** 0,91 | **Rappel (départs) :** 84 %
- Sur 38 départs réels → **32 correctement détectés**

---

## 🔑 Top 10 Variables Prédictives

| Rang | Variable | Importance |
|------|----------|-----------|
| 1 | Nb entretiens extérieurs | ★★★★★ |
| 2 | Satisfaction rémunération | ★★★★☆ |
| 3 | Charge travail perçue | ★★★★☆ |
| 4 | Satisfaction globale | ★★★★☆ |
| 5 | Nb changements d'emploi | ★★★☆☆ |
| 6 | Équilibre vie pro/perso | ★★★☆☆ |
| 7 | Promotions (3 ans) | ★★★☆☆ |
| 8 | Nb déplacements/an | ★★☆☆☆ |
| 9 | Opportunités d'évolution | ★★☆☆☆ |
| 10 | Plan de carrière défini | ★★☆☆☆ |

---

## 🗂️ Structure du projet

```
📦 Turnover_Auditeurs/
├── 📓 Serrakh_Nohaila_Madani_chaimae_CAC2.ipynb   # Notebook principal (EDA + modèles)
├── 📊 BDD_Turnover_Auditeurs.xlsx                  # Base de données (500 auditeurs, 34 variables)
├── 📄 Projet_turnover_auditeurs_Madani_Serrakh.pdf # Rapport complet
└── 📝 README.md                                    # Ce fichier
```

---

## 🚀 Lancement du notebook

### Prérequis

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl
```

### Exécution

```bash
jupyter notebook Serrakh_Nohaila_Madani_chaimae_CAC2.ipynb
```

### Librairies principales utilisées

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
```

---

## 🎬 Démonstration

📹 **Vidéo démo disponible ici :**  
[🔗 Lien Google Drive](https://drive.google.com/file/d/1BxRplw_ZEEpPJb_uMwAif_iQVuj-_Hqb/view?usp=sharing)

---

## 💡 Recommandations RH clés

- 🔔 **Alerte précoce** : scorer les auditeurs trimestriellement — seuil de risque à 65 %
- 💰 **Révision salariale** : benchmark annuel, priorité aux Directeurs et Chefs de Mission
- 📋 **Plans de développement individuels** (PDI) avec jalons de progression clairs
- 🤝 **Mentoring formalisé** : seulement 39,6 % des auditeurs en bénéficient actuellement
- 🏠 **Réduction des déplacements** pour les profils à risque + télétravail accru

---

## ⚠️ Limites

- Données **simulées** — validation sur données réelles requise avant déploiement
- Variable cible = **intention** de départ (non départ effectif)
- Pas de variables exogènes (marché de l'emploi, conjoncture économique)
- Échantillon modeste (N = 500)

---

## 📚 Références

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD'16.
- Deloitte (2023). *Global Human Capital Trends Report*.
- IBM Institute for Business Value (2022). *Employee Experience, here and now*.
- [Scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [IBM HR Analytics Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

*Projet réalisé dans le cadre du module Gestion des Ressources Humaines / Data Analytics — ENCG Settat · 2025*
