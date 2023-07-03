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

    function getTableValues($db, $tableName) {
        $query = "SELECT $tableName.id, $tableName.age_conducteur, $tableName.date_heure, $tableName.ville, $tableName.latitude, $tableName.longitude, descr_atmo.value AS descr_atmo_value, descr_dispo_secu.value AS descr_dispo_secu_value, descr_etat_surf.value AS descr_etat_surf_value, descr_lum.value AS id_descr_lum_value
            FROM $tableName
            JOIN descr_atmo ON $tableName.id_descr_atmo = descr_atmo.id
            JOIN descr_dispo_secu ON $tableName.id_descr_dispo_secu = descr_dispo_secu.id
            JOIN descr_etat_surf ON $tableName.id_descr_etat_surf = descr_etat_surf.id
            JOIN descr_lum ON $tableName.id_descr_lum = descr_lum.id";
        $stmt = $db->prepare($query);
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    function getAccidents($db) {
        $query = "SELECT *
            FROM accidents";
        $stmt = $db->prepare($query);
        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    
    


    // Handle channels request.
    $request = @$_GET['request'];
    if ($request == 'accidents'){
        $data = getTableValues($db,$request);
        header('Content-Type: application/json; charset=utf-8');
        header('Cache-control: no-store, no-cache, must-revalidate');
        header('Pragma: no-cache');
        header('HTTP/1.1 200 OK');
        echo json_encode($data);
    }

    $request = @$_GET['request'];
    if ($request == 'data_brut'){
        $data = getAccidents($db);
        header('Content-Type: application/json; charset=utf-8');
        header('Cache-control: no-store, no-cache, must-revalidate');
        header('Pragma: no-cache');
        header('HTTP/1.1 200 OK');
        echo json_encode($data);
    }

?>