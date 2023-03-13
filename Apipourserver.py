import requests
import json
payload = {
    "crVisite": "",
    "formatVisite": "presentiel",
    "nomVisite": "test Visite",
    "questionnaire": "visite",
    "lieuVisite": "",
    "data": [
        {
            "axeInvestissement": "",
            "referenceCompte": "A-00000114",
            "contacts": [
                {
                    "telephoneMobile": "",
                    "codePostal": "",
                    "ville": "",
                    "rue": "",
                    "pays": "",
                    "telephone": "",
                    "role": "rapporteur",
                    "fonction": "",
                    "prenom": "Piere",
                    "nom": "Test",
                    "email": "pierre.buttignol@numericoach.fr"
                },
                {
                    "nom": "TASSY",
                    "rue": "",
                    "ville": "",
                    "email": "atassy@amadeus-executives.com",
                    "pays": "",
                    "telephone": "0611737078",
                    "codePostal": "",
                    "role": "visiteur",
                    "telephoneMobile": "",
                    "fonction": "",
                    "prenom": "Alain"
                }
            ],
            "dureeEngagement": "",
            "styleGestion": "",
            "positionnement": "",
            "typeCompte": "EMT",
            "tailleTicket": "",
            "nomCompte": "AMADEUS EXECUTIVES"
        },
        {
            "referenceCompte": "A-00000447",
            "contacts": [
                {
                    "ville": "Paris",
                    "prenom": "Dicke 2",
                    "telephoneMobile": "",
                    "fonction": "",
                    "pays": "",
                    "telephone": "",
                    "codePostal": "",
                    "email": "dicke-balde@azed.com",
                    "rue": "9 avenue XXX",
                    "nom": "Balde",
                    "role": "prospect"
                }
            ],
            "dureeEngagement": "5 ans",
            "nomCompte": "Azed",
            "axeInvestissement": "IA",
            "tailleTicket": "1000",
            "positionnement": "Majoritaire",
            "typeCompte": "Fond",
            "styleGestion": "Actif"
        }
    ],
    "typeVisite": "1ere visite",
    "dateVisite": "05/09/2022"
}
urlAPIApex = 'https://amadeus-executives.my.salesforce.com/services/apexrest/api/v2/visites'
urlSFToken = 'https://login.salesforce.com/services/oauth2/token'
payloadtoken = {
      'client_id': '3MVG99qusVZJwhskC.5R66GUqsKD1jfke_dgKieD8tblrMUBqBPmqPQFuUnYgZF9kkR7Z2_rexsEEzjGQXWgB',    
      'client_secret': 'DB9DB4774AD895E7B18FE3C43801D6C43D55547E6EB4C95AA558E288740FD3BB',
      'grant_type': 'password',
      'username': 'apiuser@amadeus.sb1',
      'password': '2@mbkybmMTB6JTZH60S8D6sAt2E6Udt3b'
    }
res = requests.post(urlSFToken,)
print(res.text)