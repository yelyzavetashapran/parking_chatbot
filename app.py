import os
from dotenv import load_dotenv
from milvus_store import create_vector_store
from rag import create_rag_chain
import reservation
import guardrails

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in .env")
        return

    print("Initializing SQLite...")
    reservation.initialize_database()
    print("SQLite ready")

    print("Preparing Milvus vector store...")
    create_vector_store()
    print("Vector store ready")

    qa_chain = create_rag_chain()

    print("\n🤖 SmartPark Assistant ready! Type 'exit' to quit.\n")

    state = "qa"
    reservation_data = {}
    temp_proposal = None
    reservation_fields = ["first_name", "last_name", "car_number", "datetime_from", "datetime_to"]
    reservation_questions = {
        "first_name": "Enter your first name:",
        "last_name": "Enter your last name:",
        "car_number": "Enter your car number (max 8 chars):",
        "datetime_from": "Enter start datetime (YYYY-MM-DD HH:MM):",
        "datetime_to": "Enter end datetime (YYYY-MM-DD HH:MM):"
    }
    current_step = 0

    reservation_triggers = [
        "i want to reserve",
        "i want to book",
        "let's book",
        "make a reservation",
        "book a spot",
        "reserve a parking",
    ]
    info_triggers = [
        "how do i reserve",
        "reservation cost",
        "can i reserve",
        "about reservation",
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            # ---------------- INPUT GUARD ----------------
            allowed, message = guardrails.guard_input(user_input)
            if not allowed:
                print("Bot:", message)
                continue

            user_input_lower = user_input.lower()

            # ---------------- QA MODE ----------------
            if state == "qa":
                if any(trigger in user_input_lower for trigger in reservation_triggers):
                    state = "reservation_collect"
                    reservation_data = {}
                    current_step = 0
                    print("Bot: Sure! Let's make a parking reservation.")
                    print("Bot:", reservation_questions[reservation_fields[current_step]])
                    continue
                elif any(trigger in user_input_lower for trigger in info_triggers):
                    response = qa_chain.invoke({"query": user_input})
                    safe_text = guardrails.guard_output(response["result"])
                    print("Bot:", safe_text)
                    print("Bot: Do you want to make a reservation now? (yes/no)")
                    state = "reservation_offer"
                    continue
                else:
                    response = qa_chain.invoke({"query": user_input})
                    safe_text = guardrails.guard_output(response["result"])
                    print("Bot:", safe_text)
                    continue

            # ---------------- RESERVATION OFFER ----------------
            if state == "reservation_offer":
                if user_input_lower in ["yes", "y"]:
                    state = "reservation_collect"
                    reservation_data = {}
                    current_step = 0
                    print("Bot:", reservation_questions[reservation_fields[current_step]])
                elif user_input_lower in ["no", "n"]:
                    state = "qa"
                    print("Bot: Okay, let me know if you need anything else.")
                else:
                    print("Bot: Please reply 'yes' or 'no'.")
                continue

            # ---------------- RESERVATION DATA COLLECTION ----------------
            if state == "reservation_collect":
                field = reservation_fields[current_step]

                if field == "car_number":
                    valid, msg = reservation.validate_car_number(user_input)
                    if not valid:
                        print("Bot:", msg)
                        continue
                    reservation_data[field] = user_input

                elif field in ["datetime_from", "datetime_to"]:
                    valid, dt = reservation.validate_datetime(user_input)
                    if not valid:
                        print("Bot:", dt)
                        continue
                    reservation_data[field] = user_input

                    if field == "datetime_to" and "datetime_from" in reservation_data:
                        valid_from, dt_from_obj = reservation.validate_datetime(reservation_data["datetime_from"])
                        valid_to, dt_to_obj = reservation.validate_datetime(reservation_data["datetime_to"])
                        if valid_from and valid_to and dt_to_obj <= dt_from_obj:
                            print("Bot: End datetime must be after start datetime.")
                            continue
                else:
                    reservation_data[field] = user_input

                current_step += 1
                if current_step < len(reservation_fields):
                    next_field = reservation_fields[current_step]
                    print("Bot:", reservation_questions[next_field])
                    continue

                # ---------------- Create reservation proposal ----------------
                temp_proposal = reservation.create_reservation_proposal(
                    reservation_data["first_name"],
                    reservation_data["last_name"],
                    reservation_data["car_number"],
                    reservation_data["datetime_from"],
                    reservation_data["datetime_to"]
                )

                if "error" in temp_proposal:
                    message = temp_proposal["error"]
                    if "next_available_time" in temp_proposal:
                        message += f" Next available start: {temp_proposal['next_available_time']}"
                    print("Bot:", message)
                    state = "qa"
                    reservation_data = {}
                    temp_proposal = None
                    current_step = 0
                    continue

                print(f"Bot: Spot {temp_proposal['spot_number']} is available. Confirm? (yes/no)")
                state = "reservation_confirm"
                continue

            # ---------------- RESERVATION CONFIRMATION ----------------
            if state == "reservation_confirm":
                if user_input_lower in ["yes", "y"]:
                    message = reservation.confirm_reservation(temp_proposal, True)
                elif user_input_lower in ["no", "n"]:
                    message = reservation.confirm_reservation(temp_proposal, False)
                else:
                    print("Bot: Please reply 'yes' or 'no'.")
                    continue

                safe_message = guardrails.guard_output(message)
                print("Bot:", safe_message)
                state = "qa"
                reservation_data = {}
                temp_proposal = None
                current_step = 0
                continue

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Exiting.")
            break
        except Exception as e:
            print("⚠️ Error:", e)


if __name__ == "__main__":
    main()