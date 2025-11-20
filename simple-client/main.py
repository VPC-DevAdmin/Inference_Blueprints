import requests
from openai import OpenAI
import os
import httpx

BASE_URL="https://api.example.com"
KEYCLOAK_REALM="master"
KEYCLOAK_CLIENT_ID="api"
# keycloak-fetch-client-secret.sh api.example.com api-admin 'changeme!!' api
KEYCLOAK_CLIENT_SECRET="fbBPewUwz9rnEuvNyiDVgvl24mDbDE4e"

# Prepare the token endpoint and payload
token_url = f"{BASE_URL}/token"
payload = {
    "grant_type": "client_credentials",
    "client_id": KEYCLOAK_CLIENT_ID,
    "client_secret": KEYCLOAK_CLIENT_SECRET,
}

# Send the POST request
response = requests.post(token_url, data=payload, verify=False)  # verify=False skips SSL verification like -k in curl

# Check for success
if response.status_code == 200:
    access_token = response.json().get("access_token")
    print(f"Access token obtained: {access_token[:20]}...")
    
    # Use the token with OpenAI client
    # Set up OpenAI client with custom base URL and API key
    # Create httpx client with SSL verification disabled (like -k in curl)
    http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key=access_token,
        base_url=f"{BASE_URL}/CodeLlama-34b-Instruct-hf/v1",  # Match your curl endpoint
        http_client=http_client
    )
    
    # Make a sample OpenAI API call using the completions endpoint
    try:
        response = client.completions.create(
            model="codellama/CodeLlama-34b-Instruct-hf",
            prompt="Create a simple Java class that has a main method that prints 'Hello, World!'",
            max_tokens=50,
            temperature=0
        )
        print(f"Response: {response.choices[0].text}")
    except Exception as e:
        print(f"Error calling API: {e}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
