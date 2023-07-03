<?php
//Author: Prenom NOM
//Login : etuXXX
//Groupe: ISEN X GROUPE Y
//Annee:


require_once('constants.php');


function dbConnect()
{
  try
  {
    $db = new PDO('mysql:host='.DB_SERVER.';dbname='.DB_NAME.';charset=utf8',
      DB_USER, DB_PASSWORD);
    $db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION); 
  }
  catch (PDOException $exception)
  {
    error_log('Connection error: '.$exception->getMessage());
    return false;
  }
  return $db;
}


function dbAddMessage($db, $descr_dispo_secu, $descr_atmo,$descr_lum,$descr_etat_surf,$age,$date,$longitude,$latitude,$ville) {
    try {
      $request = 'INSERT INTO accidents(id_descr_dispo_secu, id_descr_atmo,id_descr_lum,id_descr_etat_surf,age_conducteur,date_heure,longitude,latitude,ville)
        VALUES(:id_descr_dispo_secu, :id_descr_atmo, :id_descr_lum, :id_descr_etat_surf, :age_conducteur, :date_heure, :longitude, :latitude, :ville)';
      $statement = $db->prepare($request);
      $statement->bindParam(':id_descr_dispo_secu', $descr_dispo_secu, PDO::PARAM_STR, 50);
      $statement->bindParam(':id_descr_atmo', $descr_atmo, PDO::PARAM_STR, 50);
      $statement->bindParam(':id_descr_lum', $descr_lum, PDO::PARAM_STR, 50);
      $statement->bindParam(':id_descr_etat_surf', $descr_etat_surf, PDO::PARAM_STR, 50);
      $statement->bindParam(':age_conducteur', $age, PDO::PARAM_STR, 3);
      $statement->bindParam(':longitude', $longitude, PDO::PARAM_STR, 4);
      $statement->bindParam(':latitude', $latitude, PDO::PARAM_STR, 4);
      $statement->bindParam(':date_heure', $date, PDO::PARAM_STR, 50);
      $statement->bindParam(':ville', $ville, PDO::PARAM_STR, 50);
      $statement->execute();
      //return dbGetMessages($db, $channelId);      
    }
    catch (PDOException $exception) {
      error_log('Request error: '.$exception->getMessage());
      var_dump($exception->getMessage());
      return false;
    }
    return true;
  }






?>
