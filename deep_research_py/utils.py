service = "openai"
model = "gpt-4o"

def get_service() -> str:
    global service
    return service

def set_service(new_service: str) -> None:
    global service
    service = new_service

def get_model() -> str:
    global model
    return model

def set_model(new_model: str) -> None:
    global model
    model = new_model
