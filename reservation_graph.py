from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
import reservation


class ReservationState(TypedDict):
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]
    car_number: Optional[str]
    datetime_from: Optional[str]
    datetime_to: Optional[str]
    proposal: Optional[dict]
    reservation_id: Optional[int]
    message: Optional[str]
    thread_id: Optional[str]
    admin_decision: Optional[str]


def create_proposal(state: ReservationState):
    proposal = reservation.create_reservation_proposal(
        state["first_name"],
        state["last_name"],
        state["email"],
        state["car_number"],
        state["datetime_from"],
        state["datetime_to"],
    )

    if "error" in proposal:
        return {"message": proposal["error"], "proposal": None}

    return {"proposal": proposal}


def route_after_proposal(state: ReservationState) -> str:
    if state.get("proposal") is None:
        return END
    return "create_pending"


def create_pending(state: ReservationState):
    reservation_id = reservation.create_pending_reservation(
        state["proposal"], state["thread_id"]
    )
    return {"reservation_id": reservation_id}


def notify_pending(state: ReservationState):
    return {"message": "Your reservation request has been sent to the administrator and is awaiting approval."}


def admin_review(state: ReservationState):
    return {}


def route_admin_decision(state: ReservationState) -> str:
    if state.get("admin_decision") == "approve":
        return "notify_approved"
    return "notify_rejected"


def notify_approved(state: ReservationState):
    from email_service import send_reservation_email
    reservation.approve_reservation(state["reservation_id"])
    info = reservation.get_reservation_email_info(state["reservation_id"])
    body = (f"Hello {info['first_name']},\n\nYour reservation has been approved.\n"
            f"Spot: {info['spot']} | {info['from']} → {info['to']}")
    send_reservation_email(info["email"], "SmartPark Reservation Approved", body)
    return {"message": "Your reservation has been approved."}


def notify_rejected(state: ReservationState):
    from email_service import send_reservation_email
    reservation.reject_reservation(state["reservation_id"])
    info = reservation.get_reservation_email_info(state["reservation_id"])
    body = (f"Hello {info['first_name']},\n\nYour reservation has been rejected. "
            f"Please call +1234567 for alternatives.")
    send_reservation_email(info["email"], "SmartPark Reservation Rejected", body)
    return {"message": "Your reservation has been rejected."}


def mcp_log(state: ReservationState):
    import requests
    from config import MCP_URL, MCP_API_KEY
    info = reservation.get_reservation_email_info(state["reservation_id"])
    try:
        requests.post(MCP_URL, headers={"X-API-KEY": MCP_API_KEY},
                      params={"first_name": info["first_name"], "last_name": info["last_name"],
                              "car_number": info["car_number"], "datetime_from": info["from"],
                              "datetime_to": info["to"]}, timeout=5)
    except Exception as e:
        print("MCP log failed:", e)
    return {"message": state.get("message")}


def build_reservation_graph(checkpointer=None):
    builder = StateGraph(ReservationState)

    builder.add_node("create_proposal", create_proposal)
    builder.add_node("create_pending", create_pending)
    builder.add_node("notify_pending", notify_pending)
    builder.add_node("admin_review", admin_review)
    builder.add_node("notify_approved", notify_approved)
    builder.add_node("mcp_log", mcp_log)
    builder.add_node("notify_rejected", notify_rejected)

    builder.set_entry_point("create_proposal")
    builder.add_conditional_edges(
        "create_proposal", route_after_proposal,
        {"create_pending": "create_pending", END: END},
    )
    builder.add_edge("create_pending", "notify_pending")
    builder.add_edge("notify_pending", "admin_review")
    builder.add_conditional_edges(
        "admin_review", route_admin_decision,
        {"notify_approved": "notify_approved", "notify_rejected": "notify_rejected"},
    )
    builder.add_edge("notify_approved", "mcp_log")
    builder.add_edge("mcp_log", END)
    builder.add_edge("notify_rejected", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["admin_review"] if checkpointer else [],
    )
