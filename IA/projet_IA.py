
# Etape 1
Découverte des données :
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import train_test_split
from sklearn import svm
from math import radians
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import json





def creer_graphiques(dictionnaire,new_equi=False):
    for cle, valeur in dictionnaire.items():
        if cle != 'Num_Acc' :
            if isinstance(valeur, dict) and 'proportion' in valeur:
                if valeur['proportion'] is not False:
                    plt.figure()
                    plt.title(cle)
                    labels = list(valeur['proportion'].keys())
                    sizes = list(valeur['proportion'].values())
                    try :
                        if new_equi== False :
                            labels = [EQUIVALANCE[cle].get(label, "Autre") for label in valeur['proportion'].keys()]
                        else :
                            labels = [NEW_EQUIVALANCE[cle].get(label, "Autre") for label in valeur['proportion'].keys()]
                    except :
                        pass
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                
    plt.show()



EQUIVALANCE={
    "descr_grav" : {1: 'Indemne', 2: 'Tué', 3: 'Blessé hospitalisé', 4: 'Blessé léger'},
    "descr_cat_veh" : {1: 'PL seul > 7,5T',2:'VU seul 1,5T <= PTAC <= 3,5T avec ou sans remorque',3: 'VL seul',4:'Autocar',5:'PL > 3,5T + remorque',6 : 'Cyclomoteur <50cm3',7:'Motocyclette > 125 cm3',8 : 'Tracteur routier + semi-remorque',9 : 'Tracteur agricole', 10 : 'PL seul 3,5T <PTCA <= 7,5T'},
    "descr_etat_surf":{1: 'Verglacé', 2:'Enneigé',3:'Mouillé',4:'Normale'},
    "descr_agglo":{1:'Hors aglomeration', 2:'En aglomeration'},
    "descr_athmo":{1:'Brouillard et fumée',2:'Neige et grele',3:'Pluie forte',4:'Normale',5:'Autre',6:'Temps éblouissant',7:'Temps couvert'},
    "descr_lum":{1 : 'Crépuscule', 2: 'Plein jour', 3 :'Nuit sans éclairage public', 4 : 'Nuit avec éclairage public'},
    "description_intersection":{1 : 'Hors intersection', 2 : 'Intersection en X', 3 : 'Giratoire', 4 : 'Intersection en T', 5 : 'Intersection à  plus de 4 branches', 6 : 'Autre intersection'},
    "descr_dispo_secu":{1:"Utilisation d'une ceinture de securité",2:"Prèsence d'une ceinture de securité - Utilisation non determminable",3:"Autre",4:"Prèsence d'une ceinture de securité non utilisée ",5:"Utilisation d'un casque"},
    "descr_motif_traj":{1:"Utilisation professionnelle", 2:"Promenade ou loisirs", 3 : 'Domicile à travail', 4 : "Domicile à école", 5 : "Courses ou achats", 6 : "Autre"},
    "descr_type_col":{1: "Deux véhicules - Frontale", 2 : "Deux véhicules par l'arrière", 3 : "Deux véhicules par le coté",4:"Sans collision",5:"Autre collision",6:"Trois véhicules et plus et plusieurs Collisions multiples",7:"Deux véhicules et Par le coté"}
    
}


NEW_EQUIVALANCE={
    "descr_grav" : {1: 'Indemne', 2: 'Blessé leger', 3 : 'Blessé grave / mort' },
    "descr_motif_traj":{1:"Autre", 2:"Promenade ou loisirs"},
    "description_intersection":{1 : 'Hors intersection', 2 : 'Intersection'},
    "descr_etat_surf":{1: 'Normale', 2:'Autre'},
    "descr_cat_veh" : {1: 'Autre',2:'VL seul'},
    "descr_agglo":{1:'Hors aglomeration', 2:'En aglomeration'},
    "descr_athmo":{1:'Autre',2:'Normale'},
    "descr_lum":{1 : 'Crépuscule', 2: 'Plein jour', 3 :'Nuit sans éclairage public', 4 : 'Nuit avec éclairage public'},
    "descr_dispo_secu":{1:"Utilisation d'une ceinture de securité",2:"Prèsence d'une ceinture de securité - Utilisation non determminable",3:"Autre"},
    "descr_type_col":{1: "Deux véhicules - Frontale", 2 : "Deux véhicules par l'arrière", 3 : "Deux véhicules par le coté",4:"Sans collision",5:"Autre collision",6:"Trois véhicules et plus et plusieurs Collisions multiples",7:"Deux véhicules et Par le coté"}
    
}    


try:
    data_big_data = pd.read_csv("ressources/export_IA.csv",delimiter=";")
except FileNotFoundError:
    print("Le fichier '{}' n'a pas été trouvé.".format("export_IA.csv"))
except Exception as e:
    print("Une erreur s'est produite :", str(e))
def decouv_donnees(df,new_equi=False):
    info_df={}

    for colonne in df.columns:
        info_colonne={}
        valeurs_uniques = df[colonne].value_counts(normalize=True).to_dict()
        nombre_valeurs_differentes = len(valeurs_uniques)
        info_colonne["valeur_diff"]=nombre_valeurs_differentes  
        if nombre_valeurs_differentes>25 :
            info_colonne["proportion"]=False
                      
        else : 
            info_colonne["proportion"]=valeurs_uniques
        
        info_df[colonne]=info_colonne
    
    creer_graphiques(info_df,new_equi)
  
decouv_donnees(data_big_data)
decouv_donnees(data_big_data,True)
Préparation des données :
# Définir la fonction de conversion
def convertir_valeur(valeur):
    return float(valeur.replace(",", ""))

# Appliquer la conversion à la colonne
data_big_data['Num_Acc'] = data_big_data['Num_Acc'].apply(convertir_valeur)
# ETAPE 2
#On regroupe les données
data_big_data['descr_grav'] = data_big_data['descr_grav'].replace({2: 3})
data_big_data['descr_grav'] = data_big_data['descr_grav'].replace({4: 2})
data_big_data['description_intersection'] = data_big_data['description_intersection'].replace({value: 2 for value in data_big_data['description_intersection'].unique() if value != 1})
data_big_data['descr_motif_traj'] = data_big_data['descr_motif_traj'].replace({value: 1 for value in data_big_data['descr_motif_traj'].unique() if value != 2})
data_big_data['descr_etat_surf'] = data_big_data['descr_etat_surf'].replace({value: 1 for value in data_big_data['descr_etat_surf'].unique() if value != 4})
data_big_data['descr_etat_surf'] = data_big_data['descr_etat_surf'].replace({4: 2})
data_big_data['descr_cat_veh'] = data_big_data['descr_cat_veh'].replace({value: 1 for value in data_big_data['descr_cat_veh'].unique() if value != 3})
data_big_data['descr_cat_veh'] = data_big_data['descr_cat_veh'].replace({3: 2})
data_big_data['descr_athmo'] = data_big_data['descr_athmo'].replace({value: 1 for value in data_big_data['descr_athmo'].unique() if value != 4})
data_big_data['descr_athmo'] = data_big_data['descr_athmo'].replace({4: 2})

data_big_data['descr_dispo_secu'] = data_big_data['descr_dispo_secu'].replace({value: 3 for value in data_big_data['descr_dispo_secu'].unique() if (value != 1 and value != 2)})
###Répartition des données
réduction manuelle
matrice_corr = data_big_data.corr(numeric_only=True)
plt.figure(figsize=(20, 8))
sns.heatmap(matrice_corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()
#print(data_big_data)
data_reduit1=data_big_data.drop(columns=['num_veh','ville','id_code_insee','Num_Acc','an_nais','descr_lum','descr_etat_surf','id_usa','place','descr_dispo_secu','date'])
matrice_corr = data_reduit1.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(matrice_corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation réduite")
plt.show()
def pourc_reduc(val_init,val_reduc) :
  pourcentage_reduc = (val_init-val_reduc)/val_init*100
  return round(pourcentage_reduc,2)
print(f'pourcentage de réduction : {pourc_reduc(len(data_big_data.columns),len(data_reduit1.columns))}%')
#ETAPE 2
df_geo = pd.DataFrame({'longitude': data_big_data['longitude'], 'latitude': data_big_data['latitude']})
df_geo_scratch = df_geo
df_geo.head()
#clustering data
n_clusters=[5, 10 , 50 , 100]
for n in n_clusters :
  kmeans = KMeans(n_clusters=n, n_init=10)
  clusters = kmeans.fit_predict(df_geo[['latitude', 'longitude']])

  df_geo['cluster'] = clusters

  centroids = kmeans.cluster_centers_

  fig = px.scatter_mapbox(df_geo, lat="latitude", lon="longitude", color="cluster", zoom=10, height=500)
  fig.add_trace(go.Scattermapbox(
      lat=centroids[:, 0], lon=centroids[:, 1],
      mode='markers', marker=dict(size=10, color='red')
  ))

  fig.update_layout(mapbox_style="open-street-map")
  fig.update_layout(title=f"Clustering des données avec K-means et scikit-learn, Nombre de clusters: {n}")

  fig.show()
#b) k-means « from scratch »
import random
def distance_l1(a, b): 
    return np.sum(np.abs(a - b))

def distance_l2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def distance_haversine(lat1, lon1, lat2, lon2):
    # Convertir les latitudes et longitudes en radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Rayon de la Terre en kilomètres
    radius = 6371.0

    # Calcul des différences de latitude et de longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Formule de Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calcul de la distance en kilomètres
    distance = radius * c

    return distance


def k_means_clustering2(data, k, distance_func, max_iterations=100):
    # Extraire les coordonnées géographiques du DataFrame
    coordinates = data[['latitude', 'longitude']].values

    # Sélectionner aléatoirement les indices initiaux des centroides
    initial_centroid_indices = random.sample(range(len(coordinates)), k)

    # Initialiser les centroides avec les coordonnées correspondantes
    centroids = coordinates[initial_centroid_indices]

    # Boucle d'itérations
    for _ in range(max_iterations):
        # Liste pour stocker les identifiants de cluster pour chaque point
        cluster_ids = []
        
        # Assigner chaque point au cluster le plus proche
        for coord in coordinates:
            if distance_func == distance_haversine:
                distances = [distance_func(coord[0], coord[1], centroid[0], centroid[1]) for centroid in centroids]
            else:
                distances = [distance_func(coord, centroid) for centroid in centroids]
            closest_cluster_id = np.argmin(distances)
            cluster_ids.append(closest_cluster_id)
        
        # Mettre à jour les coordonnées des centroides
        for cluster_id in range(k):
            cluster_coords = coordinates[np.array(cluster_ids) == cluster_id]
            centroids[cluster_id] = np.mean(cluster_coords, axis=0)
    
    # Ajouter les identifiants de cluster au DataFrame
    data['cluster'] = cluster_ids

    # Affichage des données sur une carte
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="cluster", zoom=2, height=500)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title="Clustering des données")

    # Add centroid coordinates to the DataFrame
    centroid_data = pd.DataFrame({'cluster': range(k), 'latitude': centroids[:, 0], 'longitude': centroids[:, 1]})
    data = pd.concat([data, centroid_data])
    
    # Add centroid markers
    centroid_labels = ["Centroid " + str(i) for i in range(k)]
    fig.add_trace(px.scatter_mapbox(centroid_data, lat="latitude", lon="longitude", hover_name="cluster", text=centroid_labels).data[0])

    fig.show()



def k_means_clustering(data, k, distance_func, max_iterations=100):
    # Extraire les coordonnées géographiques du DataFrame
    coordinates = data[['latitude', 'longitude']].values

    # Sélectionner aléatoirement les indices initiaux des centroides
    initial_centroid_indices = random.sample(range(len(coordinates)), k)

    # Initialiser les centroides avec les coordonnées correspondantes
    centroids = coordinates[initial_centroid_indices]

    # Boucle d'itérations
    for _ in range(max_iterations):
        # Calculer les distances de chaque point à tous les centroides
        if distance_func == distance_haversine:
            distances = np.array([[distance_func(coord[0], coord[1], centroid[0], centroid[1]) for centroid in centroids] for coord in coordinates])
        else:
            distances = np.array([[distance_func(coord, centroid) for centroid in centroids] for coord in coordinates])
        
        # Assigner chaque point au cluster le plus proche
        cluster_ids = np.argmin(distances, axis=1)

        # Mettre à jour les coordonnées des centroides
        for cluster_id in range(k):
            cluster_coords = coordinates[cluster_ids == cluster_id]
            centroids[cluster_id] = np.mean(cluster_coords, axis=0)

    # Ajouter les identifiants de cluster au DataFrame
    data['cluster'] = cluster_ids

    # Affichage des données sur une carte
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="cluster", zoom=2, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title="Clustering des données")

    # Ajouter les coordonnées des centroides au DataFrame
    centroid_data = pd.DataFrame({'cluster': range(k), 'latitude': centroids[:, 0], 'longitude': centroids[:, 1]})
    data = pd.concat([data, centroid_data])

    # Ajouter les marqueurs des centroides
    centroid_labels = ["Centroid " + str(i) for i in range(k)]
    fig.add_trace(px.scatter_mapbox(centroid_data, lat="latitude", lon="longitude", hover_name="cluster", text=centroid_labels).data[0])

    fig.show()


#On change k selon le nombre de clusters que l'on veut 
#Clusters selon la distance de Haversine
k_means_clustering(df_geo_scratch, 5, distance_func=distance_haversine, max_iterations=100)
#k_means_clustering(df_geo_scratch, 10, distance_func=distance_haversine, max_iterations=100)

#Clusters selon la distance de Manhattan (L1)
k_means_clustering(df_geo_scratch, 5, distance_func=distance_l1, max_iterations=100)
#k_means_clustering(df_geo_scratch, 10, distance_func=distance_l1, max_iterations=100)
#Clusters selon la distance euclidienne (L2)
k_means_clustering(df_geo_scratch, 5, distance_func=distance_l2, max_iterations=100)
#k_means_clustering(df_geo_scratch, 10, distance_func=distance_l2, max_iterations=100)

# Etape 3
Classification KNN
data_reduit1.head()
Test KNN avec et sans scaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json



X = data_reduit1.drop(['descr_grav'], axis=1) 
y = data_reduit1['descr_grav']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)



# Predicting on the test set
y_pred = knn_model.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Fitting the K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = knn_model.predict(X_test_scaled)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy avec scaler: {accuracy * 100:.2f}%")

Conclusion, il faut mieux utiliser le scaler
Test du k :
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json



X = data_reduit1.drop(['descr_grav'], axis=1) 
y = data_reduit1['descr_grav']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_neighbors_list=[2,5, 10, 20 , 50,75, 100, 200]

for n in n_neighbors_list :
  # Fitting the K-Nearest Neighbors model
  knn_model = KNeighborsClassifier(n_neighbors=n)
  knn_model.fit(X_train_scaled, y_train)

  # Predicting on the test set
  y_pred = knn_model.predict(X_test_scaled)

  # Calculating the accuracy of the model
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy avec scaler et n = {n}: {accuracy * 100:.2f}%")

Notre classe descr_accidents est composé de 3 labels. Notre score est > 30% donc notre modèle est viable
From scratch :
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def most_common(lst):
    return max(set(lst), key=lst.count)


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))


class KNeighborsClassifier:
    def __init__(self, k=5, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])

        return list(map(most_common, neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy



X = data_reduit1.drop(['descr_grav'], axis=1) 
y = data_reduit1['descr_grav']  


# Fusionner X et y en un seul DataFrame
data = pd.concat([X, y], axis=1)

# Prendre uniquement 1000 échantillons aléatoires
sampled_data = data.sample(n=1000, random_state=42)

# Séparer à nouveau les caractéristiques (X) et la variable cible (y) des échantillons aléatoires
X = sampled_data.drop(['descr_grav'], axis=1)
y = sampled_data['descr_grav']

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess data
ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

# Test knn model across varying ks
accuracies = []
y_pred=[]
ks = range(1, 100)
for k in ks:
    knn = KNeighborsClassifier(k=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()

print(f"Knn le plus performant est {max(accuracies)} avec un k = {accuracies.index(max(accuracies))}")
Classification de 3 fonctions de hauts niveaux
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier


# Définition de la grille de recherche pour le SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}

# Recherche des valeurs optimales pour le SVM
svm_grid_search = GridSearchCV(SVC(), svm_param_grid)
svm_grid_search.fit(X_train, y_train)

# Afficher les résultats de la recherche des valeurs optimales pour le SVM
print("Meilleurs paramètres pour SVM:", svm_grid_search.best_params_)
print("Meilleur score pour SVM:", svm_grid_search.best_score_)

# Définition de la grille de recherche pour le Random Forest
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

# Recherche des valeurs optimales pour le Random Forest
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid)
rf_grid_search.fit(X_train, y_train)

# Afficher les résultats de la recherche des valeurs optimales pour le Random Forest
print("Meilleurs paramètres pour Random Forest:", rf_grid_search.best_params_)
print("Meilleur score pour Random Forest:", rf_grid_search.best_score_)

# Définition de la grille de recherche pour le MLP
mlp_param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}

# Recherche des valeurs optimales pour le MLP
mlp_grid_search = GridSearchCV(MLPClassifier(max_iter=5000), mlp_param_grid)
mlp_grid_search.fit(X_train, y_train)

# Afficher les résultats de la recherche des valeurs optimales pour le MLP
print("Meilleurs paramètres pour MLP:", mlp_grid_search.best_params_)
print("Meilleur score pour MLP:", mlp_grid_search.best_score_)


# Création des classifieurs avec les meilleures valeurs des hyperparamètres
svm_classifier = SVC(C=svm_grid_search.best_params_['C'], gamma=svm_grid_search.best_params_['gamma'])
rf_classifier = RandomForestClassifier(n_estimators=rf_grid_search.best_params_['n_estimators'], max_depth=rf_grid_search.best_params_['max_depth'])
mlp_classifier = MLPClassifier(hidden_layer_sizes=mlp_grid_search.best_params_['hidden_layer_sizes'], alpha=mlp_grid_search.best_params_['alpha'],max_iter=5000)

# Entraînement des classifieurs
svm_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
mlp_classifier.fit(X_train, y_train)

# Prédiction sur les données de test
svm_pred = svm_classifier.predict(X_test)
rf_pred = rf_classifier.predict(X_test)
mlp_pred = mlp_classifier.predict(X_test)

# Affichage des rapports de classification pour chaque méthode
print("Rapport de classification pour SVM:")
print(classification_report(y_test, svm_pred))

print("Rapport de classification pour Random Forest:")
print(classification_report(y_test, rf_pred))

print("Rapport de classification pour MLP:")
print(classification_report(y_test, mlp_pred))
def algo_hold_out() :
    data_algo = data_reduit1.drop(columns=['descr_grav'])
    data_train, data_test, target_train, target_test = train_test_split(data_algo,data_reduit1.descr_grav,train_size=0.8)
    modele = svm.SVC()
    modele.fit(data_train, target_train)
    precision = modele.score(data_test, target_test)
    print(f'Précision du modèle :{precision}')
algo_hold_out()
from sklearn.model_selection import LeaveOneOut

#ON reduit le df a 1 000 pour aller plus vite 

data_reduit_1k = data_reduit1.sample(n=1000, random_state=42)


data_algo = data_reduit_1k.drop(columns=['descr_grav'])
etiquette = data_reduit_1k['descr_grav']
modele = svm.SVC()
scores = []

for train_index, test_index in LeaveOneOut().split(data_algo):
    data_apprentissage, data_test = data_algo.iloc[train_index], data_algo.iloc[test_index]
    etiquettes_apprentissage, etiquettes_test = etiquette.iloc[train_index], etiquette.iloc[test_index]
    modele.fit(data_apprentissage, etiquettes_apprentissage)
    score = modele.score(data_test, etiquettes_test)
    scores.append(score)

precision_moyenne = sum(scores) / len(scores)

print("Précision moyenne de l'algorithme (Leave-One-Out) :", precision_moyenne)
#La méthode Leave-One-Out permet d'utiliser tous les échantillons pour l'apprentissage,
#garantissant ainsi une utilisation maximale des données pour la modélisation. 
#Cependant, cette méthode est plus coûteuse en termes de temps de calcul. La méthode Holdout 
#est plus rapide et moins coûteuse, mais elle peut introduire un certain biais. Le choix entre 
#les deux méthodes dépend des contraintes spécifiques de votre problème, de la taille de vos 
#données et de la précision requise dans votre évaluation du modèle.


# Etape 4
Évaluation quantitative des résultats « supervisé » :
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
# Matrice de confusion pour SVM
svm_cm = confusion_matrix(y_test, svm_pred)
print("Matrice de confusion pour SVM:")
print(svm_cm)

# Matrice de confusion pour Random Forest
rf_cm = confusion_matrix(y_test, rf_pred)
print("Matrice de confusion pour Random Forest:")
print(rf_cm)

# Matrice de confusion pour MLP
mlp_cm = confusion_matrix(y_test, mlp_pred)
print("Matrice de confusion pour MLP:")
print(mlp_cm)

# Précision et rappel pour SVM
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred, average='macro')
print("Précision pour SVM:", svm_accuracy)
print("Rappel pour SVM:", svm_recall)

# Précision et rappel pour Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred, average='macro')
print("Précision pour Random Forest:", rf_accuracy)
print("Rappel pour Random Forest:", rf_recall)

# Précision et rappel pour MLP
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_recall = recall_score(y_test, mlp_pred, average='macro')
print("Précision pour MLP:", mlp_accuracy)
print("Rappel pour MLP:", mlp_recall)



# Binarisation des étiquettes
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Liste des classes à évaluer
classes_to_evaluate = [0,1,2]  # Modifier en fonction de vos classes


# Courbe ROC et AUC pour SVM
svm_prob = svm_classifier.decision_function(X_test)
fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()
for i in classes_to_evaluate:
    if y_test_bin[:, i].sum() > 0:  # Vérifier s'il y a des échantillons positifs pour la classe i
        fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test_bin[:, i], svm_prob[:, i])
        roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])
        plt.plot(fpr_svm[i], tpr_svm[i], label="SVM Class {} (AUC = {:.2f})".format(i, roc_auc_svm[i]))


# Courbe ROC et AUC pour Random Forest
rf_prob = rf_classifier.predict_proba(X_test)
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in classes_to_evaluate:
    if y_test_bin[:, i].sum() > 0:  # Vérifier s'il y a des échantillons positifs pour la classe i
        fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], rf_prob[:, i])
        roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
        plt.plot(fpr_rf[i], tpr_rf[i], label="Random Forest Class {} (AUC = {:.2f})".format(i, roc_auc_rf[i]))

# ...


# Courbe ROC et AUC pour MLP
mlp_prob = mlp_classifier.predict_proba(X_test)
fpr_mlp = dict()
tpr_mlp = dict()
roc_auc_mlp = dict()
for i in classes_to_evaluate:
    if y_test_bin[:, i].sum() > 0:  # Vérifier s'il y a des échantillons positifs pour la classe i
        fpr_mlp[i], tpr_mlp[i], _ = roc_curve(y_test_bin[:, i], mlp_prob[:, i])
        roc_auc_mlp[i] = auc(fpr_mlp[i], tpr_mlp[i])
        plt.plot(fpr_mlp[i], tpr_mlp[i], label="MLP Class {} (AUC = {:.2f})".format(i, roc_auc_mlp[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC MLP')
plt.legend(loc="lower right")
plt.show()

Évaluation quantitative des résultats « non supervisé » :
n_clusters=range(2,13)
all_silhouette_coef = []
all_calinski_harabasz_index = []
all_davies_bouldin_index = []

for n in n_clusters :
  kmeans = KMeans(n_clusters=n,n_init=10)
  cluster = kmeans.fit_predict(df_geo[['latitude', 'longitude']])
  silhouette_coef = silhouette_score((df_geo[['latitude', 'longitude']]), cluster)
  calinski_harabasz_index = calinski_harabasz_score((df_geo[['latitude', 'longitude']]), cluster)
  davies_bouldin_index = davies_bouldin_score((df_geo[['latitude', 'longitude']]), cluster)
  all_silhouette_coef.append(silhouette_coef)
  all_calinski_harabasz_index.append(calinski_harabasz_index)
  all_davies_bouldin_index.append(davies_bouldin_index)

plt.plot(n_clusters, all_silhouette_coef)
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Évolution du coefficient de silhouette en fonction du nombre de clusters')
plt.show()

plt.plot(n_clusters,all_calinski_harabasz_index)
plt.xlabel('Nombre de clusters')
plt.ylabel('calinski harabasz index')
plt.title("Évolution de l'index Calinski-Harabasz en fonction du nombre de clusters")
plt.show()

plt.plot(n_clusters,all_davies_bouldin_index)
plt.xlabel('Nombre de clusters')
plt.ylabel('davies bouldin index')
plt.title("Évolution de l'index Davies-Bouldin en fonction du nombre de clusters")
plt.show()
# ETAPE 5 
Script pour k-means :
def script_kmeans(lat,long,lst_centroids) :
  distance_min = 100000
  cluster = 0
  for centroid in lst_centroids :
    distance = distance_haversine(lat,long,centroid[0],centroid[1])
    if distance < distance_min :
      distance_min = distance
      cluster = lst_centroids.index(centroid)
  return json.dumps({'cluster':cluster})

#test fonction script_kmeans
latitude = 42.8573
longitude = 4.3485
centroids = [(48.8575, 2.3486), (48.8607, 2.3376), (48.8559, 2.3472)]

result = script_kmeans(latitude, longitude, centroids)

print(result)
Script pour KNN :
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

def knn_predict_accident(accident_info, csv_file):
    # Chargement du fichier CSV contenant la liste des accidents
    accidents_data = pd.read_csv(csv_file, delimiter=';')
    
    X = accidents_data.drop(['descr_grav'], axis=1)
    y = accidents_data['descr_grav']

    # Prétraitement des données
    # Sélection des colonnes d'informations utiles pour la prédiction
    features = accidents_data.drop(['descr_grav'], axis=1)   # Exclut la dernière colonne qui est la variable cible
    

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Création du modèle KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Remplacer 5 par le nombre de voisins souhaité

    # Entraînement du modèle
    knn_classifier.fit(X_train_scaled, y_train)

    # Prédiction sur les données de test
    y_pred = knn_classifier.predict(X_test_scaled)

    # Évaluation du modèle
    accuracy = knn_classifier.score(X_test_scaled, y_test)
    print("Précision du modèle : {:.2f}%".format(accuracy * 100))

    # Prétraitement des nouvelles données d'accident
    new_data = pd.DataFrame([accident_info], columns=features.columns)  # Modified line
    new_data_scaled = scaler.transform(new_data)

    # Prédiction pour les nouvelles données d'accident
    prediction = knn_classifier.predict(new_data_scaled)

    # Création du JSON de sortie
    output = {
        'classe_accident': prediction.tolist()
    }
    json_output = json.dumps(output)
    print("Prédiction pour les nouvelles données d'accident (JSON) :", json_output)
    return json_output


data_big_data = pd.read_csv("ressources/data_script.csv", delimiter=";")
data_reduit = data_big_data.sample(n=2000)

X = data_reduit.drop(['descr_grav'], axis=1)
y = data_reduit['descr_grav']
# Exemple d'utilisation de la fonction
accident_info = [45.6000, 5.633330, 1, 1, 1, 1, 64, 1, 1]   # Remplacer les valeurs par celles correspond
csv_file = 'ressources/data_script.csv'  # Remplacer le nom du fichier CSV
knn_predict_accident(accident_info, csv_file)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import json

data_big_data = pd.read_csv("ressources/data_script.csv", delimiter=";")
data_reduit = data_big_data.sample(n=2000)

X = data_reduit.drop(['descr_grav'], axis=1).reset_index(drop=True)
y = data_reduit['descr_grav']


# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def predict_accident(accident_info, method):
    if method == 'svm':
        svm_classifier = SVC()
        svm_classifier.fit(X_train, y_train)
        prediction = svm_classifier.predict(np.array([accident_info]))

    elif method == 'rf':
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)

        # Check if the number of features in accident_info matches X_train
        if len(accident_info) != X_train.shape[1]:
            raise ValueError("Mismatch in the number of features. Expected {} features, but got {}."
                             .format(X_train.shape[1], len(accident_info)))

        prediction = rf_classifier.predict(np.array([accident_info]))

    elif method == 'mlp':
        mlp_classifier = MLPClassifier(max_iter=5000)
        mlp_classifier.fit(X_train, y_train)
        prediction = mlp_classifier.predict(np.array([accident_info]))

    else:
        raise ValueError("Invalid method")

    output = {
        'classe_accident': prediction.tolist()
    }
    json_output = json.dumps(output)
    return json_output


# Example usage of the function
accident_info = [45.6000, 5.633330, 1, 1, 1, 1, 64, 1, 1]  # Replace with the actual values corresponding to the accident information

# Classification method to use
method = 'svm'  # Replace with the desired method ('svm', 'rf', 'mlp')

# Predict the severity of the accident
prediction = predict_accident(accident_info, method)
print("Prediction of accident severity  SVM (JSON):", prediction)

# Classification method to use
method = 'rf'  # Replace with the desired method ('svm', 'rf', 'mlp')

# Predict the severity of the accident
prediction = predict_accident(accident_info, method)
print("Prediction of accident severity Random forest (JSON):", prediction)

# Classification method to use
method = 'mlp'  # Replace with the desired method ('svm', 'rf', 'mlp')

# Predict the severity of the accident
prediction = predict_accident(accident_info, method)
print("Prediction of accident severity MLP (JSON):", prediction)
