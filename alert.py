from email.message import EmailMessage
import smtplib
def email_alert(subject, body, to):
	msg = EmailMessage()
	msg.set_content(body)
	msg['subject'] = subject
	msg['to'] = to
	user = "naitikravalafcat@gmail.com"
	msg['from'] = user
	password = "huwlprccegsoqvnj"
	server = smtplib.SMTP("smtp.gmail.com",587)
	server.starttls()
	server.login(user, password)
	server.send_message(msg)
	server.quit()
