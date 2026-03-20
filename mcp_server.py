from fastapi import FastAPI, HTTPException, Header
from datetime import datetime
from config import OUTPUT_FILE, MCP_API_KEY
from filelock import FileLock


app = FastAPI(title="SmartPark MCP Server")

LOCK_FILE = OUTPUT_FILE + ".lock"


def verify_MCP_API_KEY(x_api_key: str):
    if x_api_key != MCP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
def root():
    return {"message": "MCP Server is running"}


@app.post("/log-approved")
def log_approved_reservation(
    first_name: str,
    last_name: str,
    car_number: str,
    datetime_from: str,
    datetime_to: str,
    x_api_key: str = Header(...)
):

    verify_MCP_API_KEY(x_api_key)

    approval_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    entry = (
        f"{first_name} | {last_name} | {car_number} | "
        f"{datetime_from} -> {datetime_to} | {approval_time}\n"
    )

    try:
        lock = FileLock(LOCK_FILE)

        with lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(entry)

        return {"status": "logged"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))