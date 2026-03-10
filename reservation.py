import sqlite3
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from config import DB_FILE

TOTAL_SPOTS = 8


def _get_connection():
    return sqlite3.connect(DB_FILE)


def initialize_database():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL,
            car_number TEXT NOT NULL,
            datetime_from TEXT NOT NULL,
            datetime_to TEXT NOT NULL,
            insert_timestamp TEXT NOT NULL,
            update_timestamp TEXT NOT NULL,
            parking_spot_id TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def validate_datetime(dt_str: str) -> Tuple[bool, Any]:
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

        if dt < datetime.now():
            return False, "Datetime cannot be in the past."

        return True, dt

    except ValueError:
        return False, "Invalid datetime format. Use YYYY-MM-DD HH:MM."


def validate_car_number(car_number: str) -> Tuple[bool, Optional[str]]:
    if len(car_number) > 8:
        return False, "Car number too long (max 8 characters)."

    return True, None


def validate_email(email: str) -> Tuple[bool, Optional[str]]:

    email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

    if not re.match(email_regex, email):
        return False, "Invalid email format."

    if len(email) > 254:
        return False, "Email is too long."

    return True, None


def find_first_available_spot(start_dt: str, end_dt: str) -> Optional[str]:
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT parking_spot_id FROM reservations
        WHERE status = 'approved' AND (
            (datetime_from <= ? AND datetime_to > ?) OR
            (datetime_from < ? AND datetime_to >= ?) OR
            (datetime_from >= ? AND datetime_to <= ?)
        )
    """, (start_dt, start_dt, end_dt, end_dt, start_dt, end_dt))

    occupied = {row[0] for row in cursor.fetchall()}
    conn.close()

    for i in range(1, TOTAL_SPOTS + 1):
        spot_id = f"P{i}"
        if spot_id not in occupied:
            return spot_id

    return None


def create_reservation_proposal(
    first_name: str,
    last_name: str,
    email: str,
    car_number: str,
    datetime_from_str: str,
    datetime_to_str: str
) -> Dict[str, Any]:

    valid, msg = validate_car_number(car_number)
    if not valid:
        return {"error": msg, "retry_field": "car_number"}

    valid, dt_from = validate_datetime(datetime_from_str)
    if not valid:
        return {"error": dt_from, "retry_field": "datetime_from"}

    valid, dt_to = validate_datetime(datetime_to_str)
    if not valid:
        return {"error": dt_to, "retry_field": "datetime_to"}

    if dt_to <= dt_from:
        return {
            "error": "End datetime must be after start datetime.",
            "retry_field": "datetime_to"
        }

    spot = find_first_available_spot(datetime_from_str, datetime_to_str)

    if not spot:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT datetime_to FROM reservations
            WHERE status = 'approved'
            ORDER BY datetime_to ASC
            LIMIT 1
        """)

        result = cursor.fetchone()
        conn.close()

        next_time = result[0] if result else None

        return {
            "error": "No free spots available.",
            "next_available_time": next_time
        }

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "car_number": car_number,
        "datetime_from": datetime_from_str,
        "datetime_to": datetime_to_str,
        "spot_number": spot
    }


def create_pending_reservation(proposal: Dict[str, Any]) -> int:

    conn = _get_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    cursor.execute("""
        INSERT INTO reservations (
            first_name,
            last_name,
            email,
            car_number,
            datetime_from,
            datetime_to,
            insert_timestamp,
            update_timestamp,
            parking_spot_id,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        proposal["first_name"],
        proposal["last_name"],
        proposal["email"],
        proposal["car_number"],
        proposal["datetime_from"],
        proposal["datetime_to"],
        now_str,
        now_str,
        proposal["spot_number"],
        "pending"
    ))

    reservation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return reservation_id


def approve_reservation(reservation_id: int) -> str:
    conn = _get_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    cursor.execute("""
        UPDATE reservations
        SET status = 'approved', update_timestamp = ?
        WHERE id = ?
    """, (now_str, reservation_id))

    conn.commit()
    conn.close()

    return f"Reservation {reservation_id} approved."


def reject_reservation(reservation_id: int) -> str:
    conn = _get_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    cursor.execute("""
        UPDATE reservations
        SET status = 'rejected', update_timestamp = ?
        WHERE id = ?
    """, (now_str, reservation_id))

    conn.commit()
    conn.close()

    return f"Reservation {reservation_id} rejected."


def get_pending_reservations():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, first_name, last_name, car_number,
               datetime_from, datetime_to, parking_spot_id, status
        FROM reservations
        WHERE status = 'pending'
    """)

    rows = cursor.fetchall()
    conn.close()

    reservations = []

    for r in rows:
        reservations.append({
            "id": r[0],
            "first_name": r[1],
            "last_name": r[2],
            "car_number": r[3],
            "datetime_from": r[4],
            "datetime_to": r[5],
            "parking_spot_id": r[6],
            "status": r[7]
        })

    return reservations


def get_user_reservation_status(first_name, last_name, car_number):

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT status
        FROM reservations
        WHERE first_name = ?
        AND last_name = ?
        AND car_number = ?
        ORDER BY insert_timestamp DESC
        LIMIT 1
    """, (first_name, last_name, car_number))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return row[0]


def get_reservation_email_info(reservation_id: int):

    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT email, first_name, parking_spot_id, datetime_from, datetime_to
        FROM reservations
        WHERE id = ?
    """, (reservation_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "email": row[0],
        "name": row[1],
        "spot": row[2],
        "from": row[3],
        "to": row[4]
    }