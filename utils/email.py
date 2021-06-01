import dotenv
import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_experiment_results_email(args, results, subject):
    msg_args = str(vars(args))
    msg_results = str(results)
    msg = msg_args + "\n\n" + msg_results
    send_email(msg, subject)


def send_email(message, subject=None):
    config = dotenv.dotenv_values(".env")
    msg = MIMEText(message, "plain", "utf-8")
    my_host = config["HOST"]
    my_user = config["USER"]
    my_password = config["PASSWORD"]
    sender = config["SENDER"]
    receiver = config["RECEIVER"]

    msg["From"] = Header("道路状况预测模型", 'utf-8')
    msg["To"] = Header("憨憨调参侠", 'utf-8')
    msg["Subject"] = subject
    try:
        server = smtplib.SMTP()
        server.connect(my_host, 25)
        server.login(my_user, my_password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
    except smtplib.SMTPException:
        print("Email error.")
