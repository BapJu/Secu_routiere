import pandas as pd
import mysql.connector

# Connexion à la base de données MySQL
cnx = mysql.connector.connect(
    host='localhost',
    user='etu414',
    password='olphxwnz',
    database='etu414'
)

# Création du curseur
cursor = cnx.cursor()


nrows_full=1000
# Lecture des 100 premières lignes du fichier CSV
df = pd.read_csv('../cgi/ressources/data_to_import.csv', nrows=nrows_full)

# Parcours des lignes du DataFrame et insertion dans la base de données
for index, row in df.iterrows():
    # Récupération des valeurs des colonnes du DataFrame
    age_conducteur = row['age']
    date_heure = row['date']
    ville = row['ville']
    latitude = row['latitude']
    longitude = row['longitude']
    id_descr_etat_surf = row['descr_etat_surf']
    id_descr_atmo = row['descr_athmo']
    id_descr_dispo_secu = row['descr_dispo_secu']
    id_descr_lum = row['descr_lum']

    # Correction de la valeur de date_heure en supprimant le suffixe ' UTC'
    date_heure = date_heure.replace(' UTC', '')

    # Requête SQL pour insérer une ligne dans la table accidents
    query = "INSERT INTO accidents (age_conducteur, date_heure, ville, latitude, longitude, id_descr_etat_surf, id_descr_atmo, id_descr_dispo_secu, id_descr_lum) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"

    # Exécution de la requête SQL avec les valeurs des colonnes
    cursor.execute(query, (age_conducteur, date_heure, ville, latitude, longitude, id_descr_etat_surf, id_descr_atmo, id_descr_dispo_secu, id_descr_lum))

# Validation des modifications dans la base de données
cnx.commit()

# Fermeture du curseur et de la connexion à la base de données
cursor.close()
cnx.close()
