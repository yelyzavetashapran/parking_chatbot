from fastapi import FastAPI, HTTPException
import reservation
from graph_instance import reservation_graph


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
        thread_id = reservation.get_thread_id(reservation_id)
        if not thread_id:
            raise HTTPException(status_code=404, detail="Thread ID not found")
        config = {"configurable": {"thread_id": thread_id}}
        reservation_graph.update_state(config, {"admin_decision": "approve"}, as_node="admin_review")
        reservation_graph.invoke(None, config)
        return {"status": "approved"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reservations/{reservation_id}/reject")
def reject_reservation(reservation_id: int):
    try:
        thread_id = reservation.get_thread_id(reservation_id)
        if not thread_id:
            raise HTTPException(status_code=404, detail="Thread ID not found")
        config = {"configurable": {"thread_id": thread_id}}
        reservation_graph.update_state(config, {"admin_decision": "reject"}, as_node="admin_review")
        reservation_graph.invoke(None, config)
        return {"status": "rejected"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))