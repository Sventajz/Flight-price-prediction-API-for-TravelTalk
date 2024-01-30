from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the model
with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

class InputData(BaseModel):
    source_city: int
    destination_city: int
    airline: int
    departure_time: int
    arrival_time: int
    stops: int
    type: int

# Allow all origins in this example (you may want to restrict this in production)
origins = ["http://localhost:8080", "http://127.0.0.1:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello():
    return 'Hello, welcome to the API!'

@app.post('/predict')
async def predict(data: InputData):
    try:
        input_data = [
            [
                data.source_city,
                data.destination_city,
                data.airline,
                data.departure_time,
                data.arrival_time,
                data.stops,                
                data.type
            ]
        ]
        predictions = model.predict(input_data).tolist()  # Convert predictions to a list
        return  predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

