import requests
import pandas as pd
import json
params = {
  'access_key': "74b5ca09ab85d34f3f756c1cdbfbfd17",
  'symbols':"AAPL",
  'date_from' : '2022-12-12',
    'date_to': '2022-12-22'
}

api_result = requests.get('http://api.marketstack.com/v1/intraday', params)

api_response = api_result.json()
print(api_response)
res = pd.read(api_response,orient='split', lines=True)
print(res)