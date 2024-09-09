from fastapi import FastAPI
from enum import Enum

app = FastAPI()

class AvailableCuisines(str, Enum):
    indian = "indian"
    american = 'american'
    italian = 'italian'

food_items = {
    'indian': ['Samosa', 'Dosa'],
    'american':['Hot dog', "Apple Pie"],
    'italian': ['Ravioli', 'Pizza']
}

@app.get("/get_items/{cuisine}")

async def get_items(cuisine: AvailableCuisines):
    return food_items.get(cuisine)

