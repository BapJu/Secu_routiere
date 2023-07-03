<?php

require_once('database.php');

// Enable all warnings and errors.
ini_set('display_errors', 1);
error_reporting(E_ALL);

// Enable all warnings and errors.
ini_set('display_errors', 1);
error_reporting(E_ALL);

// Database connection.
$db = dbConnect();
if (!$db){
    header('HTTP/1.1 503 Service Unavailable');
    exit;
}

function getAccident($db,$id) {
    $query = "SELECT * FROM accidents WHERE id = $id";
    $stmt = $db->prepare($query);
    $stmt->execute();
    return $stmt->fetchAll(PDO::FETCH_ASSOC);
}




// Handle messages request.
if ($_SERVER['REQUEST_METHOD'] == 'GET') {
    if (isset($_GET['id'])) {
        $id = $_GET['id'];
        $datainit=getAccident($db, $id);
        $data=$datainit[0];
        $latitude=$data['latitude'];
        $longitude=$data['longitude'];
        $descr_atmo=$data['id_descr_atmo'];
        $descr_lum=$data['id_descr_lum'];
        $descr_etat_surf=$data['id_descr_etat_surf'];
        $descr_dispo_secu=$data['id_descr_dispo_secu'];
        $age=$data['age_conducteur'];

        // Utilisez les valeurs des variables dans la commande exec
        //$output = exec("python ../cgi/grav.py -latitude $latitude -longitude $longitude -descr_athmo $descr_atmo -descr_lum $descr_lum -descr_etat_surf $descr_etat_surf -age $age -descr_dispo_secu $descr_dispo_secu > ../cgi/logs/python_grav.txt 2>&1");
        $output = exec("python ../cgi/grav.py -latitude $latitude -longitude $longitude -descr_athmo $descr_atmo -descr_lum $descr_lum -descr_etat_surf $descr_etat_surf -age $age -descr_dispo_secu $descr_dispo_secu");
        echo $output;
    } else {
        echo 'Error';
    }
}

?>
