from pydantic import BaseModel, Field

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    agreessiveness: int = Field(
        description="How aggresive the text is on a scale from 1 to 10"
    )
    language: str = Field(description=" The language the text is written in")
