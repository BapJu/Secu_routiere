<?php
    $output = exec("python ../cgi/import_database.py > ../cgi/logs/python_database.txt 2>&1");
    echo $output;

?>
