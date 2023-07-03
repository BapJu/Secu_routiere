#install.packages("rstatix")
#install.packages("dplyr")
library(rstatix)
library(dplyr)
library(ggplot2)
library(leaflet)
library(sf)
library(scales)
library(stringr)



# Charger les données depuis Excel, Attention le fichier .R doit être lancé depuis le même dossier que l'excel
#Les données vides sont traitées comme NA
donnees <- read.csv("ressources/stat_acc_V3.csv", sep=";", header=TRUE,na.strings = "")
donnes_label <- donnees

# Recoder les catégories en nombres pour faciliter le traitement
donnees$descr_cat_veh <- as.numeric(factor(donnees$descr_cat_veh, levels = unique(donnees$descr_cat_veh)))
donnees$descr_grav <- as.numeric(factor(donnees$descr_grav, levels = unique(donnees$descr_grav)))
donnees$descr_agglo <- as.numeric(factor(donnees$descr_agglo, levels = unique(donnees$descr_agglo)))
donnees$descr_athmo <- as.numeric(factor(donnees$descr_athmo, levels = unique(donnees$descr_athmo)))
donnees$descr_lum <- as.numeric(factor(donnees$descr_lum, levels = unique(donnees$descr_lum)))
donnees$descr_etat_surf <- as.numeric(factor(donnees$descr_etat_surf, levels = unique(donnees$descr_etat_surf)))
donnees$description_intersection <- as.numeric(factor(donnees$description_intersection, levels = unique(donnees$description_intersection)))
donnees$descr_dispo_secu <- as.numeric(factor(donnees$descr_dispo_secu, levels = unique(donnees$descr_dispo_secu)))
donnees$descr_motif_traj <- as.numeric(factor(donnees$descr_motif_traj, levels = unique(donnees$descr_motif_traj)))
donnees$descr_type_col <- as.numeric(factor(donnees$descr_type_col, levels = unique(donnees$descr_type_col)))


#Converti les données date en format date
donnees$date <- as.POSIXct(donnees$date, format = "%Y-%m-%d %H:%M:%S")


#Converti les données numérique en nombre

donnees$id_usa <- as.numeric(donnees$id_usa)
donnees$id_code_insee <- sprintf("%05s", donnees$id_code_insee)
donnees$latitude <- as.numeric(donnees$latitude)
donnees$longitude <- as.numeric(donnees$longitude)
donnees$an_nais <- as.numeric(donnees$an_nais)
donnees$age <- as.numeric(donnees$age)
donnees$place <- as.numeric(donnees$place)

#On supprime les lignes contenant des NA (environ 3 000 lignes sont supprimées)
# Remplacer les valeurs supérieures à 110 par NA dans la colonne "age"
donnees$age <- ifelse(donnees$age > 110, NA, donnees$age)
donnees <- donnees[complete.cases(donnees),]
print(nrow(donnees))

#on utilise la librairie rstatix pour supprimer les données non coérantes en latitude
outliers <- identify_outliers(donnees, latitude)
outliers_long <- identify_outliers(donnees, longitude)

donnees <- anti_join(donnees, outliers)
donnees <- anti_join(donnees, outliers_long)

write_csv(donnees, "data_clean.csv")


# Agrégation par mois
monthly <- aggregate(donnees$date, by = list(YearMonth = format(donnees$date, "%m")), FUN = length)
# Agrégation par semaine
weekly <- aggregate(donnees$date, by = list(Week = format(donnees$date, "%U")), FUN = length)

# Création de la série chronologique par mois
ts_monthly <- ts(monthly$x, start = 1, frequency = 1)
# Création de la série chronologique par semaine
ts_weekly <- ts(weekly$x, start = 1, frequency = 1)

plot(ts_monthly, xlab = "Mois", ylab = "Nombre d'accidents", main = "Nombre d'accidents par mois")
plot(ts_weekly, xlab = "Semaines", ylab = "Nombre d'accidents", main = "Nombre d'accidents par semaine")


#En se basant sur ces observations, il est probablement plus approprié d'utiliser la série chronologique mensuelle pour effectuer une prévision de bonne qualité avec une régression linéaire. Les données mensuelles semblent plus stables et présentent une tendance générale, ce qui peut faciliter la modélisation et la prévision à l'aide d'une régression linéaire.

#On établi les correspondance entre le numéro de département et sa région en 2009
correspondance <- c("01"="Rhône-Alpes",
                    "02"="Picardie",
                    "03"="Auvergne",
                    "04"="Provence-Alpes-Côte d'Azur",
                    "05"="Provence-Alpes-Côte d'Azur",
                    "06"="Provence-Alpes-Côte d'Azur",
                    "07"="Rhône-Alpes",
                    "08"="Champagne-Ardenne",
                    "09"="Midi-Pyrénées",
                    "10"="Champagne-Ardenne",
                    "11"="Languedoc-Roussillon",
                    "12"="Midi-Pyrénées",
                    "13"="Provence-Alpes-Côte d'Azur",
                    "14"="Basse-Normandie",
                    "15"="Auvergne",
                    "16"="Poitou-Charentes",
                    "17"="Poitou-Charentes",
                    "18"="Centre",
                    "19"="Limousin",
                    "21"="Bourgogne",
                    "22"="Bretagne",
                    "23"="Limousin",
                    "24"="Aquitaine",
                    "25"="Franche-Comté",
                    "26"="Rhône-Alpes",
                    "27"="Haute-Normandie",
                    "28"="Centre",
                    "29"="Bretagne",
                    "2A"="Corse",
                    "2B"="Corse",
                    "30"="Languedoc-Roussillon",
                    "31"="Midi-Pyrénées",
                    "32"="Midi-Pyrénées",
                    "33"="Aquitaine",
                    "34"="Languedoc-Roussillon",
                    "35"="Bretagne",
                    "36"="Centre",
                    "37"="Centre",
                    "38"="Rhône-Alpes",
                    "39"="Franche-Comté",
                    "40"="Aquitaine",
                    "41"="Centre",
                    "42"="Rhône-Alpes",
                    "43"="Auvergne",
                    "44"="Pays de la Loire",
                    "45"="Centre",
                    "46"="Midi-Pyrénées",
                    "47"="Aquitaine",
                    "48"="Languedoc-Roussillon",
                    "49"="Pays de la Loire",
                    "50"="Basse-Normandie",
                    "51"="Champagne-Ardenne",
                    "52"="Champagne-Ardenne",
                    "53"="Pays de la Loire",
                    "54"="Lorraine",
                    "55"="Lorraine",
                    "56"="Bretagne",
                    "57"="Lorraine",
                    "58"="Bourgogne",
                    "59"="Nord-Pas-de-Calais",
                    "60"="Picardie",
                    "61"="Basse-Normandie",
                    "62"="Nord-Pas-de-Calais",
                    "63"="Auvergne",
                    "64"="Aquitaine",
                    "65"="Midi-Pyrénées",
                    "66"="Languedoc-Roussillon",
                    "67"="Alsace",
                    "68"="Alsace",
                    "69"="Rhône-Alpes",
                    "70"="Franche-Comté",
                    "71"="Bourgogne",
                    "72"="Pays de la Loire",
                    "73"="Rhône-Alpes",
                    "74"="Rhône-Alpes",
                    "75"="Île-de-France",
                    "76"="Haute-Normandie",
                    "77"="Île-de-France",
                    "78"="Île-de-France",
                    "79"="Poitou-Charentes",
                    "80"="Picardie",
                    "81"="Midi-Pyrénées",
                    "82"="Midi-Pyrénées",
                    "83"="Provence-Alpes-Côte d'Azur",
                    "84"="Provence-Alpes-Côte d'Azur",
                    "85"="Pays de la Loire",
                    "86"="Poitou-Charentes",
                    "87"="Limousin",
                    "88"="Lorraine",
                    "89"="Bourgogne",
                    "90"="Franche-Comté",
                    "91"="Île-de-France",
                    "92"="Île-de-France",
                    "93"="Île-de-France",
                    "94"="Île-de-France",
                    "95"="Île-de-France",
                    "971"="Guadeloupe",
                    "972"="Martinique",
                    "973"="Guyane",
                    "974"="La Réunion",
                    "976"="Mayotte"
)


#On récupère le nombre d'habitant pour chaque région
#source https://www.insee.fr/fr/statistiques/2119798?sommaire=2119804

correspondance_region_habitants <- c("Alsace"="1843053",
                                     "Aquitaine"="3 206 137",
                                     "Grand Est" = "1843053",
                                     "Auvergne"="1 343 964",
                                     "Basse-Normandie"="1 470 880",
                                     "Bourgogne"="1 642 440",
                                     "Bretagne"="3 175 064",
                                     "Centre"="2 538 590",
                                     "Champagne-Ardenne"="1 337 953",
                                     "Corse"="305 674",
                                     "Franche-Comté"="1 168 208",
                                     "Haute-Normandie"="1 832 942",
                                     "Île-de-France"="11 728 240",
                                     "Languedoc-Roussillon"="2 610 890",
                                     "Limousin"="741 785",
                                     "Lorraine"="2 350 112",
                                     "Midi-Pyrénées"=	"2 862 707",
                                     "Nord-Pas-de-Calais"="4 033 197",
                                     "Pays de la Loire"="3 539 048",
                                     "Picardie"="1 911 157",
                                     "Poitou-Charentes"="1 760 575",
                                     "Provence-Alpes-Côte d'Azur"="4 889 053",
                                     "Rhône-Alpes"="6 174 040",
                                     "Guadeloupe"="401 554",
                                     "Guyane"="224 469",
                                     "Martinique"="396 404",
                                     "Réunion"="816 364"
                                     )




#On attribut a chaque accident une région
for (i in 1:nrow(donnees)) {
  print(paste("Récupération des régions, ", i, "/", nrow(donnees)))
  code_insee <- donnees$id_code_insee[i]
  nombre_region <- substr(code_insee, 1, 2)
  region <- correspondance[nombre_region]
  donnees$Région[i] <-  region
  donnees$Nb_habitant[i] <- correspondance_region_habitants[region]
  print(i)
}

#On supprime les lignes contenant des NA (avec des départment inconnu exemple 97)
donnees <- donnees[complete.cases(donnees),]
print(donnees)


# Compter le nombre d'accidents par région, gravité
counts <- donnees %>%
  group_by(Région, descr_grav, Nb_habitant) %>%
  summarise(nombre_accidents = n())
  


# Utiliser la fonction mutate() pour modifier la colonne existante en supprimant les espaces en vue de convertir "les nombre" en chiffre numérique
counts <- counts %>%
  mutate(Nb_habitant = gsub("\\s", "", Nb_habitant))

counts$Nb_habitant<- as.numeric(counts$Nb_habitant)
counts$prorata <- ((counts$nombre_accidents/counts$Nb_habitant)* 100000)
#On obtient le prorata des accidents par région selon la gravité de celui-ci


###################################### VISUALISATION ######################################




######### TOTAL Région


accident_region <- counts %>%
  group_by(Région) %>%
  summarise(total_accidents = sum(nombre_accidents))

# Charger les données géographiques de la carte des régions au format GeoJSON
regions_geojson.reg <- st_read("ressources/region.geojson")

regions_geojson.reg <- regions_geojson.reg %>%
  arrange(nom)




#ON gere les exeptionsnotemment lié aux nouvelles régions
regions_geojson.reg <- regions_geojson.reg %>%
  mutate(nom = case_when(
    nom == "Auvergne-Rhône-Alpes" ~ "Auvergne",
    nom == "Bourgogne-Franche-Comté" ~ "Bourgogne",
    nom == "Centre-Val de Loire" ~ "Centre",
    nom == "Hauts-de-France" ~ "Nord-Pas-de-Calais",
    nom == "Normandie" ~ "Haute-Normandie",
    nom == "Nouvelle-Aquitaine" ~ "Aquitaine",
    nom == "Occitanie" ~ "Languedoc-Roussillon",
    nom == "Grand Est" ~ "Champagne-Ardenne",
    # Ajoutez d'autres cas spécifiques ici si nécessaire
    TRUE ~ nom
  ))



regions_geojson.reg <- regions_geojson.reg %>%
  left_join(accident_region, by=c("nom"="Région"))


ggplot() +
  geom_sf(data = regions_geojson.reg, aes(fill = total_accidents)) +
  scale_fill_gradient(low = "green", high = "red", labels = label_number(big.mark = ",")) + 
  labs(fill = "Quantités d'accidents total,\n par région en 2009") +
  theme_void()



######### PRORATA Région


accident_region <- counts %>%
  group_by(Région) %>%
  summarise(total_accidents = sum(prorata))

# Charger les données géographiques de la carte des régions au format GeoJSON
regions_geojson.reg <- st_read("ressources/region.geojson")

regions_geojson.reg <- regions_geojson.reg %>%
  arrange(nom)


#ON gere les exeptionsnotemment lié aux nouvelles régions
regions_geojson.reg <- regions_geojson.reg %>%
  mutate(nom = case_when(
    nom == "Auvergne-Rhône-Alpes" ~ "Auvergne",
    nom == "Bourgogne-Franche-Comté" ~ "Bourgogne",
    nom == "Centre-Val de Loire" ~ "Centre",
    nom == "Hauts-de-France" ~ "Nord-Pas-de-Calais",
    nom == "Normandie" ~ "Haute-Normandie",
    nom == "Nouvelle-Aquitaine" ~ "Aquitaine",
    nom == "Occitanie" ~ "Languedoc-Roussillon",
    nom == "Grand Est" ~ "Champagne-Ardenne",
    # Ajoutez d'autres cas spécifiques ici si nécessaire
    TRUE ~ nom
  ))



regions_geojson.reg <- regions_geojson.reg %>%
  left_join(accident_region, by=c("nom"="Région"))


ggplot() +
  geom_sf(data = regions_geojson.reg, aes(fill = total_accidents)) +
  scale_fill_gradient(low = "green", high = "red", labels = label_number(big.mark = ",")) + labs(fill = "Quantités d'accidents pour 400 000 habitants;\n par région en 2009") +
  theme_void()


######### TOTAL & PRORATA Départements

# Créer un tibble avec les codes de département et le nombre d'habitants

#Source insee.fr/fr/statistiques/2119792?sommaire=2119804
correspondance <- tibble(
  code = c(
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "2A",
    "2B", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "971", "972", "973", "974"
  ),
  Nb_habitant = c(
    588853, 539870, 343046, 159450, 135836, 1079100, 313578, 283296, 151117, 303298,
    353980, 277048, 1967299, 680908, 148380, 351563, 616607, 311022, 243352, 141330,
    164344, 524144, 587519, 123584, 412082, 525276, 482984, 582822, 425502, 893914,
    701883, 1230820, 187181, 1434661, 1031974, 977449, 232268, 588420, 1197038, 261277,
    379341, 327868, 746115, 223122, 1266358, 653510, 173562, 329697, 77163, 780082,
    497762, 566145, 185214, 305147, 731019, 194003, 716182, 1044898, 220199, 2571940,
    801512, 292210, 1461257, 629416, 650356, 229670, 445890, 1094439, 748614, 1708671,
    239194, 554720, 561050, 411007, 725794, 2234105, 1250120, 1313414, 1407560, 366339,
    569775, 374018, 239291, 1007303, 540065, 626411, 426066, 374849, 380192, 343377,
    142461, 1208004, 1561745, 1515983, 1318537, 1168892, 401554, 396404, 224469, 816364
  )
)




donnees$codeDep <- substr(donnees$id_code_insee, 1, 2)



donnees_regroupes <- donnees %>%
  group_by(codeDep) %>%
  summarise(total_accidents = n())


# Charger les données géographiques de la carte des régions au format GeoJSON
departement_geojson.reg <- st_read("ressources/departement.geojson")

departement_geojson.reg <- departement_geojson.reg %>%
  arrange(nom)


departement_geojson.reg <- departement_geojson.reg %>%
  left_join(donnees_regroupes, by=c("code"="codeDep"))

# Jointure avec le tableau de correspondance
departement_geojson.reg <- departement_geojson.reg %>%
  left_join(correspondance, by = c("code"))

# Afficher le tableau avec le nombre d'habitants ajouté
print(donnees)


ggplot() +
  geom_sf(data = departement_geojson.reg, aes(fill = total_accidents)) +
  scale_fill_gradient(low = "green", high = "red", labels = label_number(big.mark = ",")) + labs(fill = "Quantité d'accidents totale, \npar département en 2009") +
  theme_void()


#On attribut a chaque accident une région
for (i in 1:nrow(departement_geojson.reg)) {
  nb <- departement_geojson.reg$Nb_habitant[i]
  accident <- departement_geojson.reg$total_accidents[i]
  departement_geojson.reg$prorata[i] <- ((accident/nb)*100000)
}

ggplot() +
  geom_sf(data = departement_geojson.reg, aes(fill = prorata)) +
  scale_fill_gradient(low = "green", high = "red", labels = label_number(big.mark = ",")) + labs(fill = "Quantité d'accidents pour 100 000 habitants, \npar département en 2009") +
  theme_void()




######### TOTAL Gravité par Départements
# Compter le nombre d'accidents par région, gravité

departement_geojson.reg <- st_read("ressources/departement.geojson")

counts_2 <- donnees %>%
  group_by(codeDep, descr_grav, Nb_habitant) %>%
  summarise(nombre_accidents = n())

counts_2 <- counts_2 %>%
  mutate(Nb_habitant = gsub("\\s", "", Nb_habitant))

counts_2$Nb_habitant<- as.numeric(counts_2$Nb_habitant)

#On attribut a chaque accident une région
for (i in 1:nrow(counts_2)) {
  nb <- counts_2$Nb_habitant[i]
  accident <- counts_2$nombre_accidents[i]
  counts_2$prorata[i] <- ((accident/nb)*100000)
}

departement_geojson.reg <- departement_geojson.reg %>%
  left_join(counts_2, by = c("code"="codeDep"))

ggplot() +
  geom_sf(data = departement_geojson.reg, aes(fill = prorata)) +
  scale_fill_gradient(low = "green", high = "red", labels = label_number(big.mark = ",")) + labs(fill = "Gravité des accidents\n pour 100 000 habitants, \npar département en 2009") +
  theme_void()


################## PLOT

#REPRESENTATIONS GRAPHIQUE
library(ggplot2)

#Nombre d’accidents en fonction des conditions atmosphériques
table_athmo <- table(donnees$descr_athmo)
data_athmo <- as.data.frame(table_athmo)
names(data_athmo) <- c("conditions_athmo", "nb_acc_athmos")
diag_acc_atmo <- ggplot(data_athmo, aes(x = "", y = nb_acc_athmos, fill = conditions_athmo))+
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  scale_fill_discrete(labels = c("Brouillard", "Neige", "Pluie Forte","Normale","Autre","Temps éblouissant","Pluie légère","Temps couvert","Vent fort"))+
  labs(title = "Nombre d’accidents en fonction des conditions atmosphériques", fill = "conditions_athmo") +
  theme_void()
print(diag_acc_atmo)

#Nombre d’accidents en fonction de la description de la surface
table_surf <- table(donnees$descr_etat_surf)
data_surf <- as.data.frame(table_surf)
names(data_surf) <- c("surface", "nb_acc_surfaces")
#print(data_surf$nb_acc_surfaces)
#print(data_surf$surface)
diag_acc_surface <- ggplot(data_surf, aes(x = "", y = nb_acc_surfaces, fill = surface))+
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  scale_fill_discrete(labels = c("Autre", "Boue", "Corps Gras","Enneigée","Flaques","Innondée","Mouillée","Normale","Vergalcée"))+
  labs(title = "Nombre d’accidents en fonction de la description de la surface", fill = "surface") +
  theme_void()
print(diag_acc_surface)


data_grav <- donnees %>%
  group_by(descr_grav) %>%
  summarise(total_accidents = n())
labels_gravite <- c("Indemne", "Tué","Blessé hospitalisé", "Blessé Légé")
pie(data_grav$total_accidents, labels = labels_gravite,)
title("Nombre d’accidents selon la gravité")
#Nombre d’accidents par ville*

donnees_regroupes <- donnees %>%
  group_by(ville) %>%
  summarise(total_accidents = n())


# Trier les données par nombre d'accidents de manière croissante
donnees_triees <- donnees_regroupes[order(-donnees_regroupes$total_accidents), ]

# Sélectionner les 10 villes avec le plus grand nombre d'accidents
top_10_villes <- head(donnees_triees, 10)
# Créer le graphique à barres horizontales


# ggplot(donnees_triees, aes(x = reorder(ville, total_accidents), y = total_accidents)) +
#   geom_point(color = "blue", alpha = 0.5) 
#   labs(title = "Top 10 des villes avec le plus d'accidents en France", x = "City", y = "Number of Accidents") +
#   theme(plot.title = element_text(hjust = 0.5),
#         axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))



ggplot(top_10_villes, aes(x = reorder(ville, total_accidents), y = total_accidents)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "Top 10 des villes avec le plus d'accidents en France", x = "City", y = "Number of Accidents") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8))


#nombre d'accidents par tranches d'heure
donnees$heure <- format(donnees$date, format = "%H")
table_heure <- table(donnees$heure)
barplot(table_heure, main = "Evolution du nombre d'accidents en fonction de la tranche d'heure", xlab = "Heures", ylab = "Nombre d'accidents", col = "thistle")

#quantité d'accidents en fonction des tranches d'ages
table_age <-table(donnees$age)
barplot(table_age, main = "Evolution du nombre d'accidents en fonction de l'âge", xlab = "Âge", ylab = "Nombre d'accidents", space = 0)

#moyenne mensuelle des accidents
donnees$mois <- format(donnees$date, format = "%m") #on cree une nouvelle colonne contenant le format de date Annee-Mois
table_mois <- table(donnees$mois)
barplot(table_mois, main ="Evolution de la moyenne mensuelle des accidents\nen 2009", xlab= "Mois", ylab ="Nombre d'accidents", space = 0)


############# PARTIE ANALYSE #############


###Relation entre gravité des accidents et age

labels_gravite <- c("Indemne", "Tué","Blessé hospitalisé", "Blessé Légé")
donnees_copy <- donnees
donnees_copy$descr_grav <- factor(donnees_copy$descr_grav, levels = 1:length(labels_gravite), labels = labels_gravite)

# Discrétisation de la variable age
en2009 <- cut(donnees_copy$age, breaks = c(0, 25, 50, 75, Inf), labels = c("- 25 ans", "- 50 ans", "-75 ans", "           +75 ans"))

# Faire les tableaux croisés correspondants
tab_cross_final <- table(en2009, donnees_copy$descr_grav)

# Représenter les données avec mosaicplot
mosaicplot(tab_cross_final, color = c("aquamarine", "aquamarine4", "cornflowerblue", "cyan", "blue"), las = 2, title("Proportion de la gravité des accidents\nen France selon l'age"))


en2009 <- cut(donnees$age, breaks = c(0, 18, 25, 60, 100, Inf), labels = c("Mineurs", "- 25 ans", "- 60 ans", "Retraités", "           Centenaires"))

# Faire les tableaux croisés correspondants
tab_cross_final <- table(en2009, donnees_copy$descr_grav)

# Représenter les données avec mosaicplot
mosaicplot(tab_cross_final, color = c("aquamarine", "aquamarine4", "cornflowerblue", "cyan", "blue"), las = 2, title("Proportion de la gravité des accidents\nen France selon l'age"))

# Créer le tableau de contingence
table_age_gravite <- table(donnees$age, donnees$descr_grav)
# Effectuer le test du chi-carré
resultat_chi2 <- chisq.test(table_age_gravite)
# Afficher les résultats du test
print(resultat_chi2)


#Sur la base des résultats, on peut conclure qu'il existe une relation statistiquement significative entre l'âge et la gravité des accidents. 
#La valeur p extrêmement faible constitue une preuve solide contre l'hypothèse nulle d'indépendance, indiquant que l'âge et la gravité des accidents sont associés.


#On peut essayer pour le type de trajet :
# Créer le tableau de contingence
table_type_gravite <- table(donnees$latitude, donnees$age)
# Effectuer le test du chi-carré
resultat_chi2 <- chisq.test(table_type_gravite)
# Afficher les résultats du test
print(resultat_chi2)

# Variables à tester
variables <- c("descr_cat_veh", "descr_agglo", "descr_etat_surf", "descr_athmo","descr_lum","description_intersection","descr_dispo_secu","descr_motif_traj","descr_type_col")

# Boucle pour tester le chi-carré pour chaque variable
for (var1 in variables){
    # Créer le tableau de contingence
    table_contingence <- table(donnes_label[[var1]], donnes_label[["descr_grav"]])
    
    # Effectuer le test du chi-carré
    resultat_chi2 <- chisq.test(table_contingence)
    # Vérifier si le test est significatif
    print(resultat_chi2)
    if (resultat_chi2$p.value < 0.05) {
      cat("Variables liées :", var1, "et", "description gravité", "\n")
      cat("\n")
      title_text <- paste("Tableau croisé gravité et", var1)
      mosaicplot(table_contingence,color = c("aquamarine", "aquamarine4", "cornflowerblue", "cyan", "blue"), las = 2, title(title_text))
    }
}


###############REG
donnees_reg <- donnees
# Convertir la variable de date en mois
donnees_reg$mois <- format(donnees_reg$date, "%m")



donnees_reg <- donnees_reg %>%
  group_by(mois) %>%
  summarise(nb_accident = n())


# Calculer le nombre d'accidents par mois
donnees_par_mois <- aggregate(nb_accident ~ mois, data = donnees_reg, FUN = sum)



# Effectuer la régression linéaire
regression_mois <- lm(nb_accident ~ as.numeric(mois), data = donnees_par_mois)

# Afficher les résultats de la régression
summary(regression_mois)

# Tracer le graphique avec la droite de régression
plot(donnees_par_mois$mois, donnees_par_mois$nb_accident, xlab = "Mois", ylab = "Nombre d'accidents")
title("Regression Linéaire des accidents par mois")
abline(regression_mois, col = "red")


donnees_reg <- donnees

donnees_reg$semaine <- format(donnees_reg$date, "%W")

donnees_reg <- donnees_reg %>%
  group_by(semaine) %>%
  summarise(nb_accident = n())


# Calculer le nombre d'accidents par mois
donnees_par_semaine <- aggregate(nb_accident ~ semaine, data = donnees_reg, FUN = sum)


# Effectuer la régression linéaire
regression_semaine <- lm(nb_accident ~ as.numeric(semaine), data = donnees_par_semaine)

# Afficher les résultats de la régression
summary(regression_semaine)

# Tracer le graphique avec la droite de régression
plot(donnees_par_semaine$semaine, donnees_par_semaine$nb_accident, xlab = "Semaine", ylab = "Nombre d'accidents")
title("Regression Linéaire des accidents par semaine")
abline(regression_semaine, col = "red")

###CUMUL de la regression

cumul=0
for (i in 1:nrow(donnees_par_semaine)) {
  cumul=cumul+donnees_par_semaine$nb_accident[i]
  donnees_par_semaine$cumul_accidents[i]=cumul
}

cumul=0
for (i in 1:nrow(donnees_par_mois)) {
  cumul=cumul+donnees_par_mois$nb_accident[i]
  donnees_par_mois$cumul_accidents[i]=cumul
}

# Effectuer la régression linéaire
regression_semaine <- lm(cumul_accidents ~ as.numeric(semaine), data = donnees_par_semaine)
regression_mois <- lm(cumul_accidents ~ as.numeric(mois), data = donnees_par_mois)

# Tracer le graphique avec la droite de régression
plot(donnees_par_semaine$semaine, donnees_par_semaine$cumul_accidents, xlab = "Semaine", ylab = "Nombre d'accidents")
title("Regression Linéaire de la somme des accidents par semaine")
abline(regression_semaine, col = "red")

# Tracer le graphique avec la droite de régression
plot(donnees_par_mois$mois, donnees_par_mois$cumul_accidents, xlab = "Mois", ylab = "Nombre d'accidents")
title("Regression Linéaire de la somme des accidents par mois")
abline(regression_mois, col = "red")
summary(regression_mois)
summary(regression_semaine)

