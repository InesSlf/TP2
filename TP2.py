import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

data, meta = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data)
df["class"] = df["class"].str.decode("utf-8")
df["class"] = (df["class"] == "tested_positive").astype(int)



# X contient les caractéristiques et y contient la classe à prédire
X = df.drop("class", axis=1)
y = df["class"]

# Division du dataset : 80 % apprentissage et 20 % test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ---------- 3. Prétraitement des données ----------

# Liste des colonnes où la valeur 0 représente une valeur manquante
cols_zero = ["plas", "pres", "skin", "insu", "mass"]

# Parcours des colonnes concernées
for c in cols_zero:

    # Remplacement des zéros par NaN dans les données d'apprentissage
    X_train[c] = X_train[c].replace(0, np.nan)

    # Calcul de la médiane uniquement sur les données d'apprentissage
    median = X_train[c].median()

    # Remplacement des valeurs manquantes par la médiane
    X_train[c] = X_train[c].fillna(median)

    # Remplacement des zéros par NaN dans les données de test
    X_test[c] = X_test[c].replace(0, np.nan)

    # Utilisation de la même médiane calculée sur l'apprentissage
    X_test[c] = X_test[c].fillna(median)


#Entraînement des modèles et prédictions

# Création et entraînement du modèle GaussianNB
gnb = GaussianNB().fit(X_train, y_train)

# Prédiction des classes sur les données de test
y_pred_gnb = gnb.predict(X_test)

# Création et entraînement du modèle MultinomialNB
mnb = MultinomialNB().fit(X_train, y_train)

# Prédiction des classes sur les données de test
y_pred_mnb = mnb.predict(X_test)


#Calcul manuel de la matrice de confusion
# Fonction qui calcule les quatre valeurs de la matrice de confusion
def matrice_confusion_manuelle(y_vrai, y_pred):

    # Conversion des classes en tableaux NumPy
    y_vrai = np.array(y_vrai)
    y_pred = np.array(y_pred)

    # Comptage des vrais positifs
    VP = np.sum((y_vrai == 1) & (y_pred == 1))

    # Comptage des vrais négatifs
    VN = np.sum((y_vrai == 0) & (y_pred == 0))

    # Comptage des faux positifs
    FP = np.sum((y_vrai == 0) & (y_pred == 1))

    # Comptage des faux négatifs
    FN = np.sum((y_vrai == 1) & (y_pred == 0))

    # Retour des quatre valeurs
    return VN, FP, FN, VP


#Calcul manuel des mesures

# Fonction qui calcule les mesures d'évaluation manuellement
def mesures_manuelles(y_vrai, y_pred):

    # Récupération des valeurs de la matrice de confusion
    VN, FP, FN, VP = matrice_confusion_manuelle(y_vrai, y_pred)

    # Calcul de l'Accuracy
    accuracy = (VP + VN) / (VP + VN + FP + FN)

    # Calcul de la Precision
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0.0

    # Calcul du Recall
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0.0

    # Calcul du F1-score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Retour des quatre mesures
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


#Calcul des mesures avec sklear

# Fonction qui calcule les mêmes mesures avec sklearn
def mesures_sklearn(y_vrai, y_pred):

    return {
        "Accuracy": accuracy_score(y_vrai, y_pred),
        "Precision": precision_score(y_vrai, y_pred),
        "Recall": recall_score(y_vrai, y_pred),
        "F1-score": f1_score(y_vrai, y_pred)
    }


#Création des tableaux comparatifs

# Tableau des mesures calculées manuellement pour les deux modèles
manuel = pd.DataFrame({
    "GaussianNB": mesures_manuelles(y_test, y_pred_gnb),
    "MultinomialNB": mesures_manuelles(y_test, y_pred_mnb)
})

# Tableau des mesures calculées avec sklearn pour les deux modèles
sk = pd.DataFrame({
    "GaussianNB": mesures_sklearn(y_test, y_pred_gnb),
    "MultinomialNB": mesures_sklearn(y_test, y_pred_mnb)
})

# Affichage des mesures calculées manuellement
print("=== Mesures calculées à la main ===")
print(manuel.round(3))

# Affichage des mesures calculées avec sklearn
print("\n=== Mesures scikit-learn ===")
print(sk.round(3))

# Vérification de l'égalité entre les deux méthodes
print("\nMêmes résultats (manuel = sklearn) ?", np.allclose(manuel, sk))


#Comparaison des matrices de confusion 

# Parcours des deux modèles
for nom, pred in [
    ("GaussianNB", y_pred_gnb),
    ("MultinomialNB", y_pred_mnb)
]:

    # Calcul manuel des valeurs de la matrice
    VN, FP, FN, VP = matrice_confusion_manuelle(y_test, pred)

    # Affichage du nom du modèle
    print(f"\nMatrice de confusion {nom}")

    # Affichage de la matrice calculée manuellement
    print("Manuelle :\n", np.array([[VN, FP], [FN, VP]]))

    # Affichage de la matrice calculée avec sklearn
    print("sklearn :\n", confusion_matrix(y_test, pred))


#Graphique comparatif des mesures

# Création d'un graphique en barres
manuel.plot(kind="bar", figsize=(8, 5))

# Ajout du titre et du nom de l'axe vertical
plt.title("Comparaison GaussianNB vs MultinomialNB")
plt.ylabel("Score")

# Définition des limites de l'axe vertical
plt.ylim(0, 1)

# Affichage horizontal des noms des mesures
plt.xticks(rotation=0)

# Ajustement de la disposition du graphique
plt.tight_layout()

# Enregistrement du graphique
plt.savefig("comparaison_mesures.png")

# Affichage du graphique
plt.show()


#Graphiques des matrices de confusion

# Création d'une figure contenant deux graphiques côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Parcours des deux modèles et de leurs emplacements graphiques
for ax, pred, nom in zip(
    axes,
    [y_pred_gnb, y_pred_mnb],
    ["GaussianNB", "MultinomialNB"]
):

    # Affichage de la matrice de confusion du modèle
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred,
        display_labels=["Négatif", "Positif"],
        ax=ax,
        colorbar=False
    )

    # Ajout du titre de chaque matrice
    ax.set_title(nom)

# Ajustement de la disposition des graphiques
plt.tight_layout()

# Enregistrement des matrices de confusion
plt.savefig("matrices_confusion.png")

# Affichage des deux matrices
plt.show()
