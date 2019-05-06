import smtplib
from email.message import EmailMessage


def send_email_after_finish(func):
    def mail():
        func()
        send_mail(func_name)
    return wrapper

def send_mail(func_name)
    import smtplib

    sender = 'from@fromdomain.com'
    receivers = ['to@todomain.com']

    message = """From: From Person <from@fromdomain.com>
    To: To Person <to@todomain.com>
    Subject: SMTP e-mail test

    This is a test e-mail message.
    """

    try:
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(sender, receivers, message)         
        print "Successfully sent email"
    except SMTPException:
        print "Error: unable to send email"