import requests
import os

# bearer_token = os.getenv('BEARER_TOKEN')
bearer_token = "AAAAAAAAAAAAAAAAAAAAAMTjtwEAAAAA4SuI3E%2FNAcnrbQD9b6WZZyLHCxw%3DYTJ0En5HX6meLUAwbhBUdKNT1OS7ehUaAGy1KKq1uXQGeEMNmo"

search_url = "https://api.twitter.com/1.1/search/tweets.json"
query_params = {'query': '#bitcoin'}

def create_headers(bearer_token):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers

def connect_to_endpoint(url, headers, params):
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code} {response.text}")
    return response.json()

headers = create_headers(bearer_token)
json_response = connect_to_endpoint(search_url, headers, query_params)

print(json_response)
