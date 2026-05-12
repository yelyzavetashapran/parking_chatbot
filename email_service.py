import smtplib
from email.mime.text import MIMEText
from config import EMAIL_USER, EMAIL_PASSWORD


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def send_reservation_email(email, subject, body):

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:

        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)