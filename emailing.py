import smtplib
import imghdr
from email.message import EmailMessage


def send_email(image_path):
    email_message = EmailMessage()
    email_message["Subject"] = "New Object Detected"
    email_message.set_content("A new object has been detected.")

    with open(image_path, "rb") as file:
        content = file.read()
    email_message.add_attachment(content, maintype="image",
                                 subtype=imghdr.what(None, content))

    with open(".env", 'r') as file:
        password = file.read().split("=")[1]
    SENDER = "dannybui5155@gmail.com"
    RECEIVER = "dannybui5155@gmail.com"

    gmail = smtplib.SMTP("smtp.gmail.com", 587)
    gmail.ehlo()
    gmail.starttls()
    gmail.login(SENDER, password)

    gmail.sendmail(SENDER, RECEIVER, email_message.as_string())
    gmail.quit()
