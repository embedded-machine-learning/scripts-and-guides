#Configure Sendmail


Diesem Guide wurde gefolgt:
https://www.cloudsavvyit.com/719/how-to-set-up-a-mail-agent-for-command-line-email/


'''
sudo apt-get install postfix libsasl2-modules
cd /etc/postfix/sasl
sudo touch sasl_passwd
sudo nano sasl_passwd
    [smtp.gmail.com]:587 USERACOUNT@gmail.com:USERPASWORD
sudo postmap /etc/postfix/sasl/sasl_passwd
cd /etc/postfix/
sudo nano main.cf
'''
change relayhost to:
'''
relayhost = [smtp.gmail.com]:587
'''
add at the end
enable SASL authentication
'''
smtp_sasl_auth_enable = yes
'''
disallow methods that allow anonymous authentication:
'''
smtp_sasl_security_options = noanonymous
'''
where to find sasl_passwd:
'''
smtp_sasl_password_maps = hash:/etc/postfix/sasl/sasl_passwd
'''
Enable STARTTLS encryption:
'''
smtp_use_tls = yes
'''
Where to find CA certificates:
'''
smtp_tls_CAfile = /etc/ssl/certs/ca-certificates.crt
'''
Restart postfix:
''' 
sudo systemctl restart postfix
'''

testmail:
'''
sendmail recipient@gmail.com
FROM: USERACOUNT@gmail.com
SUBJECT: Hello from your server!
This is a test email sent from your server by Postfix.
.
'''

check for errors:
'''
sudo tail -f /var/log/mail.log
'''

