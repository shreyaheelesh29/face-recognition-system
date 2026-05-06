import requests

try:
    response = requests.get("https://api.twilio.com")
    print("Success:", response.status_code)
except Exception as e:
    print("Error:", e)
