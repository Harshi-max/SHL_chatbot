from app.models.schemas import ChatResponse


def validate_response(payload: dict) -> bool:
    ChatResponse.model_validate(payload)
    return True


if __name__ == "__main__":
    sample = {
        "reply": "Example",
        "recommendations": [],
        "end_of_conversation": False,
    }
    print(validate_response(sample))
