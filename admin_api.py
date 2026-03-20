from fastapi import FastAPI, HTTPException
import reservation
from email_service import send_reservation_email
from config import MCP_API_KEY, MCP_URL
import requests


app = FastAPI(
    title="SmartPark Admin API",
    description="API for managing parking reservations",
    version="1.0"
)


@app.get("/")
def root():
    return {"message": "SmartPark Admin API is running"}


@app.get("/admin/reservations/pending")
def get_pending_reservations():

    try:
        reservations = reservation.get_pending_reservations()
        return {"pending_reservations": reservations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reservations/{reservation_id}/approve")
def approve_reservation(reservation_id: int):

    try:
        result = reservation.approve_reservation(reservation_id)

        info = reservation.get_reservation_email_info(reservation_id)

        if info:

            body = f"""
Hello {info['first_name']},

Your parking reservation has been APPROVED.

Parking spot: {info['spot']}
From: {info['from']}
To: {info['to']}

Thank you for using SmartPark. If you have any questions, please contact admin via phone +1234567
"""

            send_reservation_email(
                info["email"],
                "SmartPark Reservation Approved",
                body
            )

            try:
                response = requests.post(
                    MCP_URL,
                    headers={
                        "X-API-KEY": MCP_API_KEY
                    },
                    params={
                        "first_name": info["first_name"],
                        "last_name": info["last_name"],
                        "car_number": info["car_number"],
                        "datetime_from": info["from"],
                        "datetime_to": info["to"]
                    },
                    timeout=5
                )

                if response.status_code != 200:
                    print("MCP error:", response.status_code, response.text)

            except Exception as e:
                print("MCP logging failed:", e)

        return {"status": "approved", "message": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reservations/{reservation_id}/reject")
def reject_reservation(reservation_id: int):

    try:
        result = reservation.reject_reservation(reservation_id)

        info = reservation.get_reservation_email_info(reservation_id)

        if info:

            body = f"""
Hello {info['first_name']},

Unfortunately your parking reservation was REJECTED.

You can create a new reservation at any time. 

If you have any questions, please contact admin via phone +1234567

SmartPark Team
"""

            send_reservation_email(
                info["email"],
                "SmartPark Reservation Rejected",
                body
            )

        return {"status": "rejected", "message": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))