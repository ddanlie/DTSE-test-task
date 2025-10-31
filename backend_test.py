from fastapi.testclient import TestClient
from app.backend import app


def test_healthcheck():
    with TestClient(app) as client:
        response = client.get("/v0/healthcheck")
        assert response.status_code == 200
        assert response.json() == {"status": "OK"}

def test_get_auth_token():
    with TestClient(app) as client:
        response = client.get("/v0/get_auth_token")
        assert response.status_code == 200
        assert "jwt_token" in response.json()

def test_predict_housing_price():
    with TestClient(app) as client:
        auth_response = client.get("/v0/get_auth_token")
        jwt_token = auth_response.json().get("jwt_token")
        
        # Prepare input data
        input_data = {
            "longitude": [-122.64, -115.73, -117.96],
            "latitude": [38.01, 33.35, 33.89],
            "housing_median_age": [36.0, 23.0, 24.0],
            "total_rooms": [1336.0, 1586.0, 1332.0],
            "total_bedrooms": [258.0, 448.0, 252.0],
            "population": [678.0, 338.0, 625.0],
            "households": [249.0, 182.0, 230.0],
            "median_income": [5.5789, 1.2132, 4.4375],
            "ocean_proximity": ["NEAR OCEAN", "INLAND", "<1H OCEAN"],
        }

        # Make the prediction request
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = client.post("/v0/predict_housing_price", json=input_data, headers=headers)
        
        results = [320201.58554044, 58815.45033765, 192575.77355635]

        assert response.status_code == 200
        assert "prediction" in response.json()

        prediction = [round(x, 8) for x in response.json()["prediction"]]
        assert prediction == results