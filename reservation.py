import sqlite3
from datetime import datetime, timedelta
from config import DB_FILE

TOTAL_SPOTS = 8

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            car_number TEXT NOT NULL,
            datetime_from TEXT NOT NULL,
            datetime_to TEXT NOT NULL,
            insert_timestamp TEXT NOT NULL,
            update_timestamp TEXT NOT NULL,
            parking_spot_id TEXT NOT NULL,
            confirmed INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def validate_datetime(dt_str):
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        if dt < datetime.now():
            return False, "Datetime cannot be in the past."
        return True, dt
    except ValueError:
        return False, "Invalid datetime format. Use YYYY-MM-DD HH:MM."

def validate_car_number(car_number):
    if len(car_number) > 8:
        return False, "Car number too long (max 8 characters)."
    return True, None


def find_first_available_spot(start_dt, end_dt):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # get all occupied spots during requested time
    cursor.execute("""
        SELECT parking_spot_id FROM reservations
        WHERE confirmed = 1 AND (
            (datetime_from <= ? AND datetime_to > ?) OR
            (datetime_from < ? AND datetime_to >= ?) OR
            (datetime_from >= ? AND datetime_to <= ?)
        )
    """, (start_dt, start_dt, end_dt, end_dt, start_dt, end_dt))
    occupied = set(r[0] for r in cursor.fetchall())
    conn.close()
    # find first free
    for i in range(1, TOTAL_SPOTS + 1):
        spot_id = f"P{i}"
        if spot_id not in occupied:
            return spot_id
    return None

def create_reservation_proposal(first_name, last_name, car_number, datetime_from_str, datetime_to_str):
    # validate car number
    valid, msg = validate_car_number(car_number)
    if not valid:
        return {"error": msg, "retry_field": "car_number"}

    # validate datetime_from
    valid, dt_from = validate_datetime(datetime_from_str)
    if not valid:
        return {"error": dt_from, "retry_field": "datetime_from"}

    # validate datetime_to
    valid, dt_to = validate_datetime(datetime_to_str)
    if not valid:
        return {"error": dt_to, "retry_field": "datetime_to"}

    if dt_to <= dt_from:
        return {"error": "End datetime must be after start datetime.", "retry_field": "datetime_to"}

    # find free spot
    spot = find_first_available_spot(datetime_from_str, datetime_to_str)
    if not spot:
        # find nearest available start
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT datetime_to FROM reservations
            WHERE confirmed = 1
            ORDER BY datetime_to ASC LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()
        next_time = result[0] if result else None
        return {"error": "No free spots available.", "next_available_time": next_time}

    return {
        "first_name": first_name,
        "last_name": last_name,
        "car_number": car_number,
        "datetime_from": datetime_from_str,
        "datetime_to": datetime_to_str,
        "spot_number": spot
    }

def confirm_reservation(proposal, confirm=True):
    if not proposal:
        return "No reservation to confirm."
    if not confirm:
        return "Reservation cancelled."

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
        INSERT INTO reservations
        (first_name, last_name, car_number, datetime_from, datetime_to,
        insert_timestamp, update_timestamp, parking_spot_id, confirmed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        proposal["first_name"],
        proposal["last_name"],
        proposal["car_number"],
        proposal["datetime_from"],
        proposal["datetime_to"],
        now_str,
        now_str,
        proposal["spot_number"],
        1
    ))
    conn.commit()
    conn.close()
    return f"Reservation confirmed. Your parking spot is {proposal['spot_number']}."