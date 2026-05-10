from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)

    @field_validator("messages")
    @classmethod
    def must_include_user_message(cls, value: list[Message]) -> list[Message]:
        if not any(msg.role == "user" for msg in value):
            raise ValueError("At least one user message is required.")
        return value


class Recommendation(BaseModel):
    name: str
    url: HttpUrl
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]


class AssessmentRecord(BaseModel):
    name: str
    url: HttpUrl
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    job_roles: list[str] = Field(default_factory=list)
    duration: str = ""
    remote_support: str = ""
    test_type: str = ""
    tags: list[str] = Field(default_factory=list)
