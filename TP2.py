import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# ---------- 1. Chargement du dataset ----------

# Fichier SMSSpamCollection : une ligne = "ham" ou "spam", une tabulation, puis le message
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df["class"] = (df["label"] == "spam").astype(int)   # spam = 1 (positif), ham = 0

# X contient les messages et y contient la classe à prédire
X = df["message"]
y = df["class"]

# Division du dataset : 80 % apprentissage et 20 % test
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)


# ---------- 2. Prétraitement : texte -> comptages de mots ----------

# Chaque message devient un vecteur contenant le nombre d'occurrences de chaque mot
# max_features limite le vocabulaire aux 1000 mots les plus fréquents
vectorizer = CountVectorizer(max_features=1000, stop_words="english")

# Le vocabulaire est appris uniquement sur les données d'apprentissage
X_train = vectorizer.fit_transform(X_train_txt)
X_test = vectorizer.transform(X_test_txt)


# ---------- 3. Entraînement des modèles et prédictions ----------

# Création et entraînement du modèle GaussianNB
# GaussianNB ne prend pas en charge les matrices creuses : on passe au format dense
gnb = GaussianNB().fit(X_train.toarray(), y_train)

# Prédiction des classes sur les données de test
y_pred_gnb = gnb.predict(X_test.toarray())

# Création et entraînement du modèle MultinomialNB (accepte directement les comptages)
mnb = MultinomialNB().fit(X_train, y_train)

# Prédiction des classes sur les données de test
y_pred_mnb = mnb.predict(X_test)


# ---------- 4. Calcul manuel de la matrice de confusion ----------

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


# ---------- 5. Calcul manuel des mesures ----------

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


# ---------- 6. Calcul des mesures avec scikit-learn ----------

# Fonction qui calcule les mêmes mesures avec sklearn
def mesures_sklearn(y_vrai, y_pred):

    return {
        "Accuracy": accuracy_score(y_vrai, y_pred),
        "Precision": precision_score(y_vrai, y_pred),
        "Recall": recall_score(y_vrai, y_pred),
        "F1-score": f1_score(y_vrai, y_pred)
    }


# ---------- 7. Création des tableaux comparatifs ----------

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


# ---------- 8. Comparaison des matrices de confusion ----------

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


# ---------- 9. Graphique comparatif des mesures ----------

# Création d'un graphique en barres
manuel.plot(kind="bar", figsize=(8, 5))

# Ajout du titre et du nom de l'axe vertical
plt.title("Comparaison GaussianNB vs MultinomialNB (SMS Spam)")
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


# ---------- 10. Graphiques des matrices de confusion ----------

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
        display_labels=["Ham", "Spam"],
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