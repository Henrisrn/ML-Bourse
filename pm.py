import requests
import json
import time
while(True):
    url = "https://login.salesforce.com/services/oauth2/token"

    payload={      
            'client_id': '3MVG99qusVZJwhskC.5R66GUqsKD1jfke_dgKieD8tblrMUBqBPmqPQFuUnYgZF9kkR7Z2_rexsEEzjGQXWgB',    
        'client_secret': 'DB9DB4774AD895E7B18FE3C43801D6C43D55547E6EB4C95AA558E288740FD3BB',
        'grant_type': 'password',
        'username': 'apiuser@amadeus.sb1',
        'password': '2@mbkybmMTB6JTZH60S8D6sAt2E6Udt3b'}
    files=[

    ]
    headers = {
    'Cookie': 'BrowserId=uMQVO3lhEe2NzD2TvFr1Zw; CookieConsentPolicy=0:0; LSKey-c$CookieConsentPolicy=0:0'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    res = json.loads(str(response.text))
    print(res["access_token"])

    url = "https://amadeus-executives.my.salesforce.com/services/apexrest/api/v2/visites"

    payload = "{\r\n    \"crVisite\": \"\",\r\n    \"formatVisite\": \"presentiel\",\r\n    \"nomVisite\": \"test Visite\",\r\n    \"questionnaire\": \"visite\",\r\n    \"lieuVisite\": \"\",\r\n    \"data\": [\r\n        {\r\n            \"axeInvestissement\": \"\",\r\n            \"referenceCompte\": \"A-00000114\",\r\n            \"contacts\": [\r\n                {\r\n                    \"telephoneMobile\": \"\",\r\n                    \"codePostal\": \"\",\r\n                    \"ville\": \"\",\r\n                    \"rue\": \"\",\r\n                    \"pays\": \"\",\r\n                    \"telephone\": \"\",\r\n                    \"role\": \"rapporteur\",\r\n                    \"fonction\": \"\",\r\n                    \"prenom\": \"Piere\",\r\n                    \"nom\": \"Test\",\r\n                    \"email\": \"pierre.buttignol@numericoach.fr\"\r\n                },\r\n                {\r\n                    \"nom\": \"TASSY\",\r\n                    \"rue\": \"\",\r\n                    \"ville\": \"\",\r\n                    \"email\": \"atassy@amadeus-executives.com\",\r\n                    \"pays\": \"\",\r\n                    \"telephone\": \"0611737078\",\r\n                    \"codePostal\": \"\",\r\n                    \"role\": \"visiteur\",\r\n                    \"telephoneMobile\": \"\",\r\n                    \"fonction\": \"\",\r\n                    \"prenom\": \"Alain\"\r\n                }\r\n            ],\r\n            \"dureeEngagement\": \"\",\r\n            \"styleGestion\": \"\",\r\n            \"positionnement\": \"\",\r\n            \"typeCompte\": \"EMT\",\r\n            \"tailleTicket\": \"\",\r\n            \"nomCompte\": \"AMADEUS EXECUTIVES\"\r\n        },\r\n        {\r\n            \"referenceCompte\": \"A-00000447\",\r\n            \"contacts\": [\r\n                {\r\n                    \"ville\": \"Paris\",\r\n                    \"prenom\": \"Dicke 2\",\r\n                    \"telephoneMobile\": \"\",\r\n                    \"fonction\": \"\",\r\n                    \"pays\": \"\",\r\n                    \"telephone\": \"\",\r\n                    \"codePostal\": \"\",\r\n                    \"email\": \"dicke-balde@azed.com\",\r\n                    \"rue\": \"9 avenue XXX\",\r\n                    \"nom\": \"Balde\",\r\n                    \"role\": \"prospect\"\r\n                }\r\n            ],\r\n            \"dureeEngagement\": \"5 ans\",\r\n            \"nomCompte\": \"Azed\",\r\n            \"axeInvestissement\": \"IA\",\r\n            \"tailleTicket\": \"1000\",\r\n            \"positionnement\": \"Majoritaire\",\r\n            \"typeCompte\": \"Fond\",\r\n            \"styleGestion\": \"Actif\"\r\n        }\r\n    ],\r\n    \"typeVisite\": \"1ere visite\",\r\n    \"dateVisite\": \"05/09/2022\"\r\n}"
    headers = {
    'Authorization': 'Bearer '+res["access_token"],
    'Content-Type': 'text/plain',
    'Cookie': 'BrowserId=uMQVO3lhEe2NzD2TvFr1Zw; CookieConsentPolicy=0:1; LSKey-c$CookieConsentPolicy=0:1'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    time.sleep(15)