import requests 
host = 'churn-serving-env.eba-zpjawnyp.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict'
# Test data for the loaded model
customers = [#Json
    {
        "lead_source": "organic_search",
        "industry": "technology",
        "employment_status": "employed",
        "location": "europe",
        "annual_income": 100160,
        "number_of_courses_viewed": 13,
        "interaction_count": 4,
        "lead_score": 0.84
    },
]
response = requests.post(
    url,
    json=customers[0],
    timeout=10
)

response.raise_for_status()
#response = response.json()

response.raise_for_status()

prediction = response.json()

print("HTTP Status:", response.status_code)
print("URL:", response.url)
print("Prediction:", prediction)


print("HTTP Status:", response.status_code)
print("URL:", response.url)
print("Prediction:", response.json())

#print(response)

