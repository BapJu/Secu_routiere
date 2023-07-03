<?php

    // Appel de la fonction Python avec le paramètre d'entrée via exec()
    //$output=exec("python ../cgi/knn2.py $latitude $longitude");
    //echo $output."----";


    require_once('constants.php');


    // Créer une connexion
    $conn = new mysqli(DB_SERVER, DB_USER, DB_PASSWORD, DB_NAME);

    // Vérifier la connexion
    if ($conn->connect_error) {
        die("La connexion a échoué : " . $conn->connect_error);
    }

    // Nom de la table dont vous voulez récupérer les en-têtes de colonnes
    $tableName = "accidents";

    // Exécuter la requête pour récupérer les en-têtes de colonnes
    $sql = "SHOW COLUMNS FROM $tableName";
    $result = $conn->query($sql);

    if ($result->num_rows > 0) {
        // Ouvrir le fichier CSV en écriture
        $file = fopen('../cgi/ressources/data_script.csv', 'w');

        // Récupérer les noms des colonnes
        $columnNames = array();
        while ($row = $result->fetch_assoc()) {
            $columnNames[] = $row['Field'];
        }

        // Écrire les en-têtes des colonnes dans le fichier CSV
        fputcsv($file, $columnNames);

        // Exécuter une autre requête pour récupérer les données de la table
        $dataSql = "SELECT * FROM $tableName";
        $dataResult = $conn->query($dataSql);

        // Parcourir chaque ligne de résultat
        while ($row = $dataResult->fetch_assoc()) {
            // Écrire chaque ligne de résultat dans le fichier CSV
            fputcsv($file, $row);
        }

        // Fermer le fichier CSV
        fclose($file);
    } else {
        echo "Aucun résultat trouvé dans la base de données.";
    }

    // Fermer la connexion à la base de données
    $conn->close();

    $output = exec("python ../cgi/knn.py > ../cgi/logs/python_log.txt 2>&1");

    $output = exec("python ../cgi/knn.py");
    //$output = exec("../cgi/knn.py");
    echo $output;


?>