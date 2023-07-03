import argparse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import json

def KNN(info_acc):
    data = pd.read_csv(fichier_csv, delimiter=';')
    data=data.sample(n=5000)

    # Séparation des données en base d'apprentissage et de test
    X = data.drop("descr_grav", axis=1)
    y = data['descr_grav']

    # Apprentissage
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(X, y)

    # Préparation des données d'entrée pour la prédiction
    info_acc_df = pd.DataFrame([info_acc])

    # Prédiction
    prediction = neigh.predict(info_acc_df)
    correlation = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
    reponse = correlation[prediction[0]]

    # Calcul de l'accuracy score
    y_pred = neigh.predict(X)
    accuracy = accuracy_score(y, y_pred)
    result={"acc_score":accuracy,"pred":reponse}
    return result


def SVM(info_acc):
    data = pd.read_csv(fichier_csv, delimiter=';')
    data=data.sample(n=5000)
    # Séparation des données en base d'apprentissage et de test
    X = data.drop("descr_grav", axis=1)
    y = data['descr_grav']

    # Apprentissage
    svm = SVC()
    svm.fit(X, y)

    # Préparation des données d'entrée pour la prédiction
    info_acc_df = pd.DataFrame([info_acc])

    # Prédiction
    prediction = svm.predict(info_acc_df)
    correlation = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
    reponse = correlation[prediction[0]]

    # Calcul de l'accuracy score
    y_pred = svm.predict(X)
    accuracy = accuracy_score(y, y_pred)
    result={"acc_score":accuracy,"pred":reponse}
    return result

from sklearn.model_selection import train_test_split

def random_forest(info_acc):
    data = pd.read_csv(fichier_csv, delimiter=';')

    # Séparation des données en base d'apprentissage et de test
    X = data.drop("descr_grav", axis=1)
    y = data['descr_grav']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apprentissage
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    # Préparation des données d'entrée pour la prédiction
    info_acc_df = pd.DataFrame([info_acc])

    # Prédiction
    prediction = random_forest.predict(info_acc_df)
    correlation = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
    reponse = correlation[prediction[0]]

    # Calcul de l'accuracy score sur l'ensemble de test
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    result = {"acc_score": accuracy, "pred": reponse}

    return result


def MLP(info_acc):
    data = pd.read_csv(fichier_csv, delimiter=';')

    # Séparation des données en base d'apprentissage et de test
    X = data.drop("descr_grav", axis=1)
    y = data['descr_grav']

    # Apprentissage
    mlp = MLPClassifier()
    mlp.fit(X, y)

    # Préparation des données d'entrée pour la prédiction
    info_acc_df = pd.DataFrame([info_acc])

    # Prédiction
    prediction = mlp.predict(info_acc_df)
    correlation = {1: "Indemne", 2: "Tué", 3: "Blessé hospitalisé", 4: "Blessé léger"}
    reponse = correlation[prediction[0]]

    # Calcul de l'accuracy score
    y_pred = mlp.predict(X)
    accuracy = accuracy_score(y, y_pred)
    result={"acc_score":accuracy,"pred":reponse}
    return result


def checkArguments():
    """Check program arguments and return program parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-age', '--age', type=int, required=True, help='age')
    parser.add_argument('-latitude', '--latitude', type=float, required=True,
                        help='latitude')
    parser.add_argument('-longitude', '--longitude', type=float, required=True,
                        help='longitude')
    parser.add_argument('-descr_lum', '--descr_lum', type=int, required=True,
                        help='luminosity')
    parser.add_argument('-descr_athmo', '--descr_athmo', type=int, required=True,
                        help='atmospheric')
    parser.add_argument('-descr_etat_surf', '--descr_etat_surf', type=int, required=True,
                        help='surface')
    parser.add_argument('-descr_dispo_secu', '--descr_dispo_secu', type=int, required=True,
                        help='security')
    return parser.parse_args()




args = checkArguments()
data = {
 'latitude' : args.latitude,
 'longitude': args.longitude,
 'descr_athmo': args.descr_athmo,
 'descr_lum': args.descr_lum,
 'descr_etat_surf': args.descr_etat_surf,
 'age':args.age,
 'descr_dispo_secu': args.descr_dispo_secu,
}

fichier_csv='../cgi/ressources/data_clean.csv'
response={}
response['KNN']=KNN(data)
response['SVM']=SVM(data)
response['random_forest']=random_forest(data)
response['MLP']=MLP(data)
json_data = json.dumps(response)
print(json_data)
