import os
from dotenv import load_dotenv

from milvus_store import create_vector_store
from rag import create_rag_chain
from reservation_graph import build_reservation_graph

import reservation
import guardrails


RESERVATION_FIELDS = [
    "first_name",
    "last_name",
    "email",
    "car_number",
    "datetime_from",
    "datetime_to",
]

STATUS_FIELDS = [
    "first_name",
    "last_name",
    "car_number",
]

RESERVATION_QUESTIONS = {
    "first_name": "Enter your first name:",
    "last_name": "Enter your last name:",
    "email": "Enter your email address:",
    "car_number": "Enter your car number (max 8 chars):",
    "datetime_from": "Enter start datetime (YYYY-MM-DD HH:MM):",
    "datetime_to": "Enter end datetime (YYYY-MM-DD HH:MM):",
}

STATUS_QUESTIONS = {
    "first_name": "Enter your first name:",
    "last_name": "Enter your last name:",
    "car_number": "Enter your car number:",
}

RESERVATION_TRIGGERS = [
    "i want to reserve",
    "i want to book",
    "let's book",
    "make a reservation",
    "book a spot",
    "reserve a parking",
]

STATUS_TRIGGERS = [
    "reservation status",
    "status of my reservation",
    "check my reservation",
    "my reservation status",
]

INFO_TRIGGERS = [
    "how do i reserve",
    "reservation cost",
    "can i reserve",
    "about reservation",
]


def initialize_system():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in .env")
        return None, None

    print("Initializing SQLite...")
    reservation.initialize_database()
    print("SQLite ready")

    print("Preparing Milvus vector store...")
    create_vector_store()
    print("Vector store ready")

    qa_chain = create_rag_chain()
    reservation_graph = build_reservation_graph()

    return qa_chain, reservation_graph


def run_qa(qa_chain, user_input):
    response = qa_chain.invoke({"query": user_input})
    safe_text = guardrails.guard_output(response["result"])
    print("Bot:", safe_text)


def reset_reservation_state():
    return {}, None, 0


def ask_next_question(step, fields, questions):
    field = fields[step]
    print("Bot:", questions[field])


def main():

    qa_chain, reservation_graph = initialize_system()

    if not qa_chain:
        return

    print("\n🤖 SmartPark Assistant ready! Type 'exit' to quit.\n")

    state = "qa"
    reservation_data = {}
    current_step = 0

    while True:

        try:

            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            user_input_lower = user_input.lower()

            if state == "qa" and any(trigger in user_input_lower for trigger in STATUS_TRIGGERS):

                state = "reservation_status"
                reservation_data, _, current_step = reset_reservation_state()

                print("Bot: Let's check your reservation status.")
                ask_next_question(current_step, STATUS_FIELDS, STATUS_QUESTIONS)

                continue

            allowed, message = guardrails.guard_input(user_input)

            if not allowed:
                print("Bot:", message)
                continue


            if state == "qa":

                if any(trigger in user_input_lower for trigger in RESERVATION_TRIGGERS):

                    state = "reservation_collect"
                    reservation_data, _, current_step = reset_reservation_state()

                    print("Bot: Sure! Let's make a parking reservation.")
                    ask_next_question(current_step, RESERVATION_FIELDS, RESERVATION_QUESTIONS)

                    continue

                if any(trigger in user_input_lower for trigger in INFO_TRIGGERS):

                    run_qa(qa_chain, user_input)

                    print("Bot: Do you want to make a reservation now? (yes/no)")
                    state = "reservation_offer"

                    continue

                run_qa(qa_chain, user_input)
                continue


            if state == "reservation_offer":

                if user_input_lower in ["yes", "y"]:

                    state = "reservation_collect"
                    reservation_data, _, current_step = reset_reservation_state()
                    ask_next_question(current_step, RESERVATION_FIELDS, RESERVATION_QUESTIONS)

                elif user_input_lower in ["no", "n"]:

                    state = "qa"
                    print("Bot: Okay, let me know if you need anything else.")

                else:

                    print("Bot: Please reply 'yes' or 'no'.")

                continue

            if state == "reservation_collect":

                field = RESERVATION_FIELDS[current_step]

                if field == "email":

                    valid, msg = reservation.validate_email(user_input)

                    if not valid:
                        print("Bot:", msg)
                        continue    
                    
                if field == "car_number":

                    valid, msg = reservation.validate_car_number(user_input)

                    if not valid:
                        print("Bot:", msg)
                        continue

                elif field in ["datetime_from", "datetime_to"]:

                    valid, result = reservation.validate_datetime(user_input)

                    if not valid:
                        print("Bot:", result)
                        continue

                reservation_data[field] = user_input

                current_step += 1

                if current_step < len(RESERVATION_FIELDS):

                    ask_next_question(current_step, RESERVATION_FIELDS, RESERVATION_QUESTIONS)
                    continue


                result = reservation_graph.invoke({
                    "first_name": reservation_data["first_name"],
                    "last_name": reservation_data["last_name"],
                    "email": reservation_data["email"],
                    "car_number": reservation_data["car_number"],
                    "datetime_from": reservation_data["datetime_from"],
                    "datetime_to": reservation_data["datetime_to"],
                })

                message = guardrails.guard_output(result.get("message", "Reservation processed."))

                print("Bot:", message)

                state = "qa"
                reservation_data, _, current_step = reset_reservation_state()
                continue


            if state == "reservation_status":

                field = STATUS_FIELDS[current_step]
                reservation_data[field] = user_input

                current_step += 1

                if current_step < len(STATUS_FIELDS):

                    ask_next_question(current_step, STATUS_FIELDS, STATUS_QUESTIONS)
                    continue

                status = reservation.get_user_reservation_status(
                    reservation_data["first_name"],
                    reservation_data["last_name"],
                    reservation_data["car_number"]
                )

                if not status:

                    print("Bot: I couldn't find your reservation.")

                elif status == "pending":

                    print("Bot: Your reservation is still pending administrator approval.")

                elif status == "approved":

                    print("Bot: Your reservation has been approved.")

                elif status == "rejected":

                    print("Bot: Unfortunately your reservation was rejected.")

                state = "qa"
                reservation_data, _, current_step = reset_reservation_state()

        except KeyboardInterrupt:

            print("\n👋 Interrupted. Exiting.")
            break

        except Exception as e:

            print("⚠️ Error:", e)


if __name__ == "__main__":
    main()