from pydantic import BaseModel
from typing import List

class Card(BaseModel):
    title: str
    image: str
    snippet: str
    deep_link: str  

class AIResponse(BaseModel):
    answer: str
    cards: List[Card]