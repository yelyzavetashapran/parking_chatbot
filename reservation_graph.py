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
        return {"message": proposal["error"]}

    return {"proposal": proposal}


def create_pending(state: ReservationState):

    proposal = state["proposal"]
    reservation_id = reservation.create_pending_reservation(proposal)

    return {"reservation_id": reservation_id}


def send_to_admin(state: ReservationState):
    return {"message": 'Your reservation request has been sent to the administrator and is awaiting approval.'}


def build_reservation_graph():

    builder = StateGraph(ReservationState)

    builder.add_node("create_proposal", create_proposal)
    builder.add_node("create_pending", create_pending)
    builder.add_node("send_to_admin", send_to_admin)

    builder.set_entry_point("create_proposal")

    builder.add_edge("create_proposal", "create_pending")
    builder.add_edge("create_pending", "send_to_admin")
    builder.add_edge("send_to_admin", END)

    return builder.compile()