from fastapi import Header, HTTPException

API_KEY_CREDENTIAL = "sk_test_123456789" # In production, use .env

async def validate_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY_CREDENTIAL:
        raise HTTPException(
            status_code=401, 
            detail={"status": "error", "message": "Invalid API key or malformed request"}
        )
    return x_api_key