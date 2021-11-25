<div align="center">
  <img src="../_img/eml_logo_and_text.png">
</div>

# Configure Sendmail

Create a gmail account and disable two factor authentification. 

**Do not use your personal gmail account because you will save the acount name and password in clear text on the device.**

You can also follow this Guide:

 *[Setup mail agent for command line](https://www.cloudsavvyit.com/719/how-to-set-up-a-mail-agent-for-command-line-email/) 

## Install needed Modules:
```
sudo apt-get install postfix libsasl2-modules
```

During installation choos **Internet Site** as 'General Type of mail configuration':

## Create Configuration File:
```
cd /etc/postfix/sasl
sudo touch sasl_passwd
```
## Fill in Configuartion File:
```
sudo nano sasl_passwd
[smtp.gmail.com]:587 USERACOUNT@gmail.com:USERPASWORD
sudo postmap /etc/postfix/sasl/sasl_passwd
```
## Edit Postfix Configuration File:
```
cd /etc/postfix/
sudo nano main.cf
```
Change relayhost to:
```
relayhost = [smtp.gmail.com]:587
```
Add at the end of main.cf:

Enable SASL authentication:
```
smtp_sasl_auth_enable = yes
```
Disallow methods that allow anonymous authentication:
```
smtp_sasl_security_options = noanonymous
```
where to find sasl_passwd:
```
smtp_sasl_password_maps = hash:/etc/postfix/sasl/sasl_passwd
```
Enable STARTTLS encryption:
```
smtp_use_tls = yes
```
Where to find CA certificates:
```
smtp_tls_CAfile = /etc/ssl/certs/ca-certificates.crt
```
# Restart postfix:
``` 
sudo systemctl restart postfix
```

# Send a Testmail:
```
sendmail recipient@gmail.com
FROM: USERACOUNT@gmail.com
SUBJECT: Hello from your server!
This is a test email sent from your server by Postfix.
.
```

# check for errors:
```
sudo tail -f /var/log/mail.log
```

