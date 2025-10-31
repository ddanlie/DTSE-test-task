import os
import pandas as pd
import json
import jwt
from typing import Annotated

import datetime
import logging
from dotenv import load_dotenv
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Request, HTTPException, Body, Depends
from fastapi.params import Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from main import load_model, predict, transform_input_data

# Load environment
#load_dotenv(dotenv_path=Path(__file__).parent.resolve() / "env" / ".env")
JWT_SECRET_KEY = os.getenv("JWT_SECRET", "default_secret_key")

# Set up logging
LOGFILE_PATH = Path(__file__).parent.resolve() / "api.log"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(LOGFILE_PATH, mode='w')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: [%(funcName)s]----->%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Long lived objects
ml_models = {
    "price_prediction_model": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ML models are loading...")
    ml_models["price_prediction_model"] = load_model("model.joblib")
    logger.info("ML models loaded and ready to use")
    yield
    ml_models.clear()

# Dependencies
auth_scheme = HTTPBearer()
def verify_jwt_token(credentials: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)]):
    logger.info("Verifying JWT token...")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=["HS256"])
        logger.info("JWT token is valid")
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid JWT token")
        raise HTTPException(status_code=401, detail="Invalid token")

# Endpoints
api_router = APIRouter()

@api_router.get("/healthcheck")
def healthcheck() -> JSONResponse:
    logger.info("Healthcheck endpoint called")
    if ml_models["price_prediction_model"] is None:
        err_msg = "Healthcheck failed: Price prediction model is not loaded"
        logger.error(err_msg)
        return JSONResponse(content={"status": err_msg}, status_code=500)
    logger.info("Healthcheck succeeded")
    return JSONResponse(content={"status": "OK"})

@api_router.get("/get_auth_token")
def get_auth_token() -> JSONResponse:
    jwt_token = jwt.encode({
        "sub": "",
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=3000),
        "iat": datetime.datetime.now(datetime.timezone.utc),
    }, JWT_SECRET_KEY, algorithm="HS256")
    content={
        "jwt_token": jwt_token
    }
    return JSONResponse(content=content)

@api_router.post("/predict_housing_price", dependencies=[Depends(verify_jwt_token)])
async def predict_housing_price(data: dict = Body(...)) -> JSONResponse: # type: ignore
    logger.info("Prediction of housing price requested")
    # Check if all fields are the same length
    if not all(len(vector) == len(data.get("longitude", [])) for vector in data.values()):
        err_msg = "Invalid input data: wrong length"
        logger.error(err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    logger.info(f"Input data length: {len(data.get('longitude', []))}")
    raw_X = {
        "longitude"         : data.get("longitude"),
        "latitude"          : data.get("latitude"),
        "housing_median_age": data.get("housing_median_age"),
        "total_rooms"       : data.get("total_rooms"),
        "total_bedrooms"    : data.get("total_bedrooms"),
        "population"        : data.get("population"),
        "households"        : data.get("households"),
        "median_income"     : data.get("median_income"),
        "ocean_proximity"   : data.get("ocean_proximity")
    }
    if any(value is None for value in raw_X.values()):
        err_msg = "Invalid input data: Missing fields"
        logger.error(err_msg)
        raise HTTPException(status_code=400, detail=err_msg)
    logger.info(f"Input data is valid, predicting...")
    X = transform_input_data(pd.DataFrame(raw_X))
    Y = predict(X, ml_models["price_prediction_model"])
    return JSONResponse(content={"prediction": Y.tolist()})

# FastAPI app and router
app = FastAPI(lifespan=lifespan)
app.include_router(api_router,  prefix="/v0")
logger.info("FastAPI app is initialized, routers included")