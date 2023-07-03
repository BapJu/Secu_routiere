import pandas as pd
from sklearn.cluster import KMeans
import os
import json



def create_cluster_list(data_file, n_clusters):
    df = pd.read_csv(data_file, encoding='latin-1')
    df_geo = pd.DataFrame({'longitude': df['longitude'], 'latitude': df['latitude']})
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    
    
    clusters = kmeans.fit_predict(df_geo[['latitude', 'longitude']])
    df_geo['cluster'] = clusters
    
    cluster_list = []
    for cluster_id in range(n_clusters):
        cluster_data = df_geo[df_geo['cluster'] == cluster_id][['latitude', 'longitude']].values.tolist()
        cluster_list.append(cluster_data)
    return cluster_list

# Exemple d'utilisation de la fonction
cluster_list = create_cluster_list("../cgi/ressources/data_script.csv", 5)
dic_json = json.dumps(cluster_list)
print(dic_json)

