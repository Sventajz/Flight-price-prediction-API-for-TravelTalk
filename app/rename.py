from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model
with open('./model.joblib', 'rb') as f:
    model = joblib.load(f)

class InputData(BaseModel):
    airline: int
    flight: int
    source_city: int
    departure_time: int
    stops: int
    arrival_time: int
    destination_city: int
    duration: float
    days_left: int
    type:int

@app.get('/')
def hello():
    return 'Hello, welcome to the API!'

@app.post('/predict')
async def predict(data: InputData):
    try:
        input_data = [
            [
                data.airline,
                data.flight,
                data.source_city,
                data.departure_time,
                data.stops,
                data.arrival_time,
                data.destination_city,
                data.duration,
                data.days_left,
		        data.type
            ]
        ]
        predictions = model.predict(input_data).tolist()  # Convert predictions to a list
        return {'predictions': predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
