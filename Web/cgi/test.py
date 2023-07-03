#!/usr/bin/python3.9

# exemple d'appel du script en ligne de commande ou exec php : python /var/www/etu000/cgi/test.py request=bonjour
# exemple d'appel du script depuis le navigateur http://etu000.projets.isen-ouest.fr/cgi/test.py?request=bonjour

# ecrit le header requis pour un affichage dans le navigateur (ne pas oublier le retour à la ligne obligatoire en fin de print) 
print ("Content-type: application/json \r\n")

# https://docs.python.org/3.9/library/cgi.html
import cgi
import cgitb

# https://www.w3schools.com/python/python_json.asp
import json

# activation debeug CGI
cgitb.enable()

# recurperation des parametres de l'url envoyés par le formulaire dans la variable form
form = cgi.FieldStorage()

# verification de présence d'une clé dans le tableau form
if 'request' not in form:
    print('request attribute required !')
    exit()
    request = 'ERREUR'
else:
    # lecture de la valeur de l'attribut dans une variable
    request = form['request'].value

# la variable request doit normalement contenir la valeur fournie en argument lors de l'appel du script
# on l'utilise ensuite pour realiser un affichage de démo:

# variables de démo pour la transformation d'un dictionnaire et d'un tableau python en JSON
tab = ['v1','v2','v3']
dic = {'request': request, 'tab': tab}

# transformation de la variable dic en chaine de caractères JSON
dic_json = json.dumps(dic)

# affichage d'exemple d'une chaine json
# print ('{"request": "'+request+'", "tab": ["v1","v2","v3"]}')
print(dic_json)

