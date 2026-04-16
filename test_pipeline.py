import uuid, pytest, threading
from unittest.mock import patch
from langgraph.checkpoint.memory import InMemorySaver
from reservation_graph import build_reservation_graph
import reservation

PROPOSAL = {
    "first_name": "Alice", "last_name": "Smith", "email": "a@b.com",
    "car_number": "ABC123", "datetime_from": "2099-01-01 10:00",
    "datetime_to": "2099-01-01 12:00", "spot_number": "P1"
}

def make_graph():
    return build_reservation_graph(InMemorySaver())

def make_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_proposal_error_stops_graph():
    graph = make_graph()
    with patch("reservation.create_reservation_proposal", return_value={"error": "no spots"}):
        result = graph.invoke(
            {"first_name": "A", "last_name": "B", "email": "a@b.com",
             "car_number": "X", "datetime_from": "d", "datetime_to": "d",
             "thread_id": "t1", "admin_decision": None},
            make_config()
        )
    assert "no spots" in result["message"]


def test_hitl_approve_flow():
    graph = make_graph()
    config = make_config()
    with patch("reservation.create_reservation_proposal", return_value=PROPOSAL), \
         patch("reservation.create_pending_reservation", return_value=42), \
         patch("reservation.approve_reservation"), \
         patch("reservation.get_reservation_email_info", return_value={
             "email": "a@b.com", "first_name": "Alice", "last_name": "Smith",
             "car_number": "ABC123", "spot": "P1", "from": "2099-01-01 10:00", "to": "2099-01-01 12:00"
         }), \
         patch("email_service.send_reservation_email") as mock_email, \
         patch("requests.post") as mock_mcp:
        graph.invoke({"first_name": "Alice", "last_name": "Smith", "email": "a@b.com",
                      "car_number": "ABC123", "datetime_from": "2099-01-01 10:00",
                      "datetime_to": "2099-01-01 12:00", "thread_id": "t2", "admin_decision": None}, config)
        graph.update_state(config, {"admin_decision": "approve"}, as_node="admin_review")
        graph.invoke(None, config)
        assert mock_email.called
        assert mock_mcp.called


def test_hitl_reject_flow():
    graph = make_graph()
    config = make_config()
    with patch("reservation.create_reservation_proposal", return_value=PROPOSAL), \
         patch("reservation.create_pending_reservation", return_value=43), \
         patch("reservation.reject_reservation"), \
         patch("reservation.get_reservation_email_info", return_value={
             "email": "a@b.com", "first_name": "Alice", "last_name": "Smith",
             "car_number": "ABC123", "spot": "P1", "from": "2099-01-01 10:00", "to": "2099-01-01 12:00"
         }), \
         patch("email_service.send_reservation_email") as mock_email, \
         patch("requests.post") as mock_mcp:
        graph.invoke({"first_name": "Alice", "last_name": "Smith", "email": "a@b.com",
                      "car_number": "ABC123", "datetime_from": "2099-01-01 10:00",
                      "datetime_to": "2099-01-01 12:00", "thread_id": "t3", "admin_decision": None}, config)
        graph.update_state(config, {"admin_decision": "reject"}, as_node="admin_review")
        graph.invoke(None, config)
        assert mock_email.called
        assert not mock_mcp.called


def test_get_thread_id_round_trip():
    import os
    test_db = "data/test_tmp.db"
    with patch("reservation.DB_FILE", test_db):
        reservation.initialize_database()
        tid = str(uuid.uuid4())
        rid = reservation.create_pending_reservation(PROPOSAL, tid)
        assert reservation.get_thread_id(rid) == tid
    os.remove(test_db)


def test_concurrent_reservations():
    errors = []
    def run():
        try:
            graph = make_graph()
            config = make_config()
            with patch("reservation.create_reservation_proposal", return_value=PROPOSAL), \
                 patch("reservation.create_pending_reservation", return_value=1):
                graph.invoke({"first_name": "A", "last_name": "B", "email": "a@b.com",
                              "car_number": "X", "datetime_from": "d", "datetime_to": "d",
                              "thread_id": config["configurable"]["thread_id"], "admin_decision": None},
                             config)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)
    assert not errors
