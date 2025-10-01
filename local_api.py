import requests

BASE = "http://127.0.0.1:8000"

# GET /
r = requests.get(f"{BASE}/")
print("Status Code:", r.status_code)
print("Result:", r.json().get("message"))

# POST /data/
sample = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}
r = requests.post(f"{BASE}/data/", json=sample)
print("Status Code:", r.status_code)
try:
    print("Result:", r.json().get("result"))
except Exception:
    print("Body:", r.text)
