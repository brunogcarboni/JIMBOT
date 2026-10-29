
---


# Accessing the SSH of the AntMiner Board and the Raspbarry Pi 3

The aproach we took was connecting the AntMiner board directly with the Ethernet cable input of the PC while it's configured to share it's connection with any wired device. If you don't know how to do it, follow the instructions below; if it is not the case, you can skip them.

1. Access the `Configuration` app in Ubuntu and click on `Network`.
2. Click on the gear icon to access `Network Settings`.
3. There, in the `IPv4` tab, select `Shared with other computers` and click `Apply`.

The Raspbarry Pi was connected to the same wifi the PC is connected.

Obs.: Please, use Linux.


## Antminer's SSH
    
Install the network mapping tool 'nmap' by running:
    
```sh
sudo apt install nmap
```
    
After the installation, run the following command:

```sh
ip -4 route | grep 'proto kernel
```

Look for the DHCP's ip created by linux (usually 10.42.0.X/X) and then run:

```shell
sudo nmap -sn 10.42.0.X/X
```

_Obs.: Replace X/X by the IP and range you found and it's corresponding range, if you didn't take the same aproach as we did and connected the Board to a router, the IP and range is usually 192.168.X.X/XX_

Find the right device and it's IP and access the SSH by running:
    
```shell
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa root@10.42.0.X
```

> [!NOTE]
> Because of the elder operating system Bitmain on the board, Linux has some troble connecting with the board. The flags '+diffie-hellman-group14-sha1' and '+ssh-rsa' are needed for a SSH connection.

After accessing the SSH, it will ask something and will await for a input, type 'yes'.
After typing, the command line will ask for a password. The default password fof the Bitmain facotry firmware is usuaaly 'admin', if it doesn't work, try typing 'root'.
    
Once averything go well, you should see the following text in you terminal.
    
```sh
root@antMiner:~#
```
    
## Raspbarry Pi's SSH 

---

# Communication between the Antiminer Control Board and the Raspbarry Pi 3

## Setting up the physical connection

If you look closely to the Antiminer Control Board you can see tree pins near the boot configuration jumpers, the microSD Card slot and the Ethernet module. The slot is labeled J12 and the pins are GND,TXD and RXD.

### Connecting the wires to the Rasbarry Pi 3

Antminer's GND -> Rasbarry Pi's GND (Physical pin 6)

Antminer's TXD -> Rasbarry Pi's GPIO14 (RX)(Physical pin 8)

Antminer's RXD -> Rasbarry Pi's GPIO15 (TX)(Physical pin 10)


## Setting up some configurations

### Enabling Serial Port on Raspbarry Pi 3

Type `sudo raspi-config` in the terminal and follow the instructions:

1. Go to `Interface Options`
2. Select `Serial Port`

There, you should disable the login **shell over serial** and enable **hardware serial interface**.

### UART Clock Drift problem (Disabling Bluetooth)

Because the default port's baud reate fluctuated with CPU clock changes (causing framing errors and data corruption), Blutooth must be disabled.

Acces the Pi's boot configuration file by running `sudo nano /boot/firmware/config.txt` or `sudo nano /boot/config.txt` depending on the OS version.

Add the following line into the file:

```shell
dtoverlay=disable-bt
```
To verify if everything went well, run:

```shell
ls -l /dev/serial0
```
The output should show the following:

```sh
lrwxrwxrwx 1 root root 5 Oct 29 01:46 /dev/serial0 -> ttyAMA0
```

### Granting permanent read/write access to the serial interface

Write this command:

```shell
sudo usermod -a -G dialout $USER
```

This command is needed otherwise your user won't have permission to write data to the serial interface and **permission denied** like errors will show.