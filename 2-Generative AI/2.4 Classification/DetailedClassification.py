from pydantic import BaseModel, Field

class DetailedClassification(BaseModel):
    sentiment: str = Field(
        description="The sentiment of the text", 
        enum=["happy", "neutral", "sad", "angry"]
    )
    agreessiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        description=" The language the text is written in",
        enum=["spanish", "english", "french", "german", "italian"]
        )
