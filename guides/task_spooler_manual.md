<div align="center">
  <img src="../_img/eml_logo_and_text.png">
</div>

# Task Spooler Manual for EDA-Server

## Verbindung zu EDA-Servern:

Ich verweiße hier auf [https://phabricator.ict.tuwien.ac.at/w/ict_servicesandinfrastructure/ict_eda_server/]  wo alle Informationen bezüglich Verbindung zu den EDA-Servern erklärt
werden.

Bevorzugt sollen die folgenden zwei Server Verwendet werden:

  1. eda01.ict.tuwien.ac.at (128.131.80.56) - GPU - Unterstützt
  2. eda02.ict.tuwien.ac.at (128.131.80.57) - Rein CPU



## Verwendung von Task-Spooler

Grundsätzlich wird Task Spooler mittels tsp aufgerufen. Die
Dokumentation ist unter [http://manpages.ubuntu.com/manpages/xenial/man1/tsp.1.html]
Es ist ts durch tsp zu ersetzen.

### Start von Task-Spooler

Es muss die Richtige Socket verwendet werden, damit alle Programme in
einer Reihe liegen und der Reihe nach ausgeführt werden. Im zweiten
Schritt wird der Pfad für das Outputfile erstellt. Entweder im
Standardpfad oder in einem Userpfad PATH-OUTPUTFILE.out ist nur ein
generischer Platzhalter bitte euren Pfad einfügen.

```
1a. export TS_SOCKET="/srv/ts_socket/CPU.socket"
1b. export TS_SOCKET="/srv/ts_socket/GPU.socket"
2. tsp
3a. chmod 777 /srv/ts_socket/CPU.socket
3b.chmod 777 /srv/ts_socket/GPU.socket
4a. export TS_TMPDIR="/srv/ts_socket/Output/" (Standardpfad)
4b. export TS_TMPDIR="/PATH-OUTPUTFILE.out" (Userpfad optional)

```
### Wichtig:
Der Export muss auserhalb einer Virtuellen umgebung (pip-enviroment) passieren.


### Start eines Scripts
```
tsp  -L LABEL-FOR-SCRIPT /PATH-TO-SCRIPT
```


### Fortschritt
```
tsp
oder tsp -l
```
### Zeigen des gerade laufenden Outputfiles
```
tsp -c <id>
```

falls der Prozess noch läuft, wird die Konsolenansicht mit str+c beendet.


### Vertauschen in der Queue
```
tsp -U <id-id>
```

### Töten eines Prozesses:
```
kill $(tsp - p <id>)
```
### Entfernen eines Prozesses:
```
tsp -r <id>
```

### Queue aufräumen:
Löschen von fertigen Jobs.
```
tsp -C
```

### Übertragen von Dateien:

Senden:
```
scp Quelldatei.bsp Benutzer@Host:Verzeichnis/Zieldatei.bsp
```
Empfangen:
```
scp Benutzer@Host:Verzeichnis/Quelldatei.bsp Zieldatei.bsp
```

## Beispiele:

Unter /srv/ts_socket/Testrun befindet sich ein Beispielscript (Workscript.sh)

Beispielscript 1
```
ssh USERNAME@eda01.ict.tuwien.ac.at
export TS_SOCKET="/srv/ts_socket/CPU.socket"
export TS_TMPDIR="/srv/ts_socket/Output/"
tsp
chmod 777 /srv/ts_socket/CPU.socket
tsp -L "USERNAME" /srv/ts_socket/Testrun/Workscript.sh

```

Beispielscript 2
```
#!/bin/bash

echo "Start script with . /[PATH_TO_SCRIPT], not with ./"

NAME=XXX

echo "=== Init task spooler ==="
echo "Setup task spooler socket for GPU."

export TS_SOCKET="/srv/ts_socket/GPU.socket"
chmod 777 /srv/ts_socket/GPU.socket
export TS_TMPDIR=/home/$NAME/logs
echo task spooler output directory: /home/$NAME/logs

```

## Geplante Ordnerstruktur:

Für jeden Benutzer wird unter /srv/cdl-eml/User/USERNAME ein
Benutzer erstellt. Darin sollen dann alle von ihm gewünschten virtuellen
Enviroments mit eigener Namensgebung angelegt werden. Für die
Outputfiles kann dann innerhalb dieses Ordners entweder eine eigene
Ordnerstruktur angelegt werden oder /srv/cdl-eml/Output verwendet
werden.


