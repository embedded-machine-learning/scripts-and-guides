<div align="center">
  <img src="../_img/eml_logo_and_text.png">
</div>

# Task Spooler Manual for EDA-Server

ich habe es jetzt auf allen Jetson Geräten ausgetestet - es funktioniert auf allen gleich:
Zum Maximieren der clocks reicht folgender Scriptaufruf:
```
sudo /usr/bin/jetson_clocks
```

du könntest zusätzlich nach deinem Script die alten clocks wiederherstellen wenn du 

```
sudo /usr/bin/jetson_clocks --restore 
```

ausführst.
(Dafür musst du allerdings vorher die default-Werte mit 

```
sudo /usr/bin/jetson_clocks --store 
```

abspeichern.)
Ansehen kann man die aktuelle clock config mit

```
sudo /usr/bin/jetson_clocks --show 
```

## Setup Ohne Sudo
```
sudo visudo
```

im öffnenden sudoers file folgende Zeile (nach %sudo   ALL=(ALL:ALL) ALL) einfügen:

```
cdleml  ALL=(ALL) NOPASSWD: /usr/bin/jetson_clocks
```

Das bedeutet, dass der user cdleml das jetson_clock script als sudo ohne Passwort ausführen darf.  
-> damit sollte der Aufruf aus deinem Pythonscript ohne Passwort und ohne root-Rechte funktionieren!
(Die scripts, die man ins sudoers file einträgt, sollten alle im ownership von root sein, da sonst normale user beliebigen code als sudo ausführen können.)



