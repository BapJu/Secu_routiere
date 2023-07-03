<?php
//Author: Prenom NOM
//Login : etuXXX
//Groupe: ISEN X GROUPE Y
//Annee:


    require_once('database.php');

    // Enable all warnings and errors.
    ini_set('display_errors', 1);
    error_reporting(E_ALL);

    // Database connection.
    $db = dbConnect();
    if (!$db)
    {
    header('HTTP/1.1 503 Service Unavailable');
    exit;
    }

    // Votre code de connexion à la base de données ici
    // ...

    // Fonction pour récupérer les valeurs d'une table et les renvoyer sous forme de tableau
    function getTableValues($db, $tableName) {
        $query = "SELECT * FROM $tableName";
        $stmt = $db->prepare($query);
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    // Récupération des valeurs des tables
    

    if ($db) {
        $descr_atmo = getTableValues($db, 'descr_atmo');
        $descr_dispo_secu = getTableValues($db, 'descr_dispo_secu');
        $descr_etat_surf = getTableValues($db, 'descr_etat_surf');
        $descr_lum = getTableValues($db, 'descr_lum');
    } else {
        // Gestion de l'erreur de connexion à la base de données
        header('HTTP/1.1 503 Service Unavailable');
        exit;
    }

    // Renvoi des données au format JSON
    $data = array(
        'descr_atmo' => $descr_atmo,
        'descr_dispo_secu' => $descr_dispo_secu,
        'descr_etat_surf' => $descr_etat_surf,
        'descr_lum' => $descr_lum
    );

    header('Content-Type: application/json; charset=utf-8');
    header('Cache-control: no-store, no-cache, must-revalidate');
    header('Pragma: no-cache');
    header('HTTP/1.1 200 OK');
    echo json_encode($data);



    // Handle messages request.
    if ($_SERVER['REQUEST_METHOD'] == 'POST') {
        if (isset($_POST['descr_dispo_secu']) && isset($_POST['descr_atmo']) && isset($_POST['descr_lum']) && isset($_POST['descr_etat_surf']) && isset($_POST['age']) && isset($_POST['date']) && isset($_POST['longitude']) && isset($_POST['latitude'])&& isset($_POST['ville'])){
        $data = dbAddMessage($db, $_POST['descr_dispo_secu'], $_POST['descr_atmo'], $_POST['descr_lum'], $_POST['descr_etat_surf'], $_POST['age'], $_POST['date'], $_POST['longitude'], $_POST['latitude'],$_POST['ville']);
        } else {
        echo 'Error';
        }
    }
    
    


?>
