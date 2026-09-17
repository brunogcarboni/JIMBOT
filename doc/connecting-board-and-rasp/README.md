1. Access the SSH of the AntMiner Board and the Raspbarry Pi 3

The aproach we took was connecting the raspbarry to the same wifi the PC is connected and the connecteting the AntMiner board directly with the Ethernet cable input of the PC while it's configured to share it's connection with any wired device.

Obs.: Please, use Linux.

- Antminer's SSH
    
Install the network mapping tool 'nmap' by running:
    
'''sh
sudo apt install nmap
'''
    
After the installation, run the following command:

'''sh
ip -4 route | grep 'proto kernel
'''

Look for the DHCP's ip created by linux (usually 10.42.0.X/X) and then run:

'''sh
sudo nmap -sn 10.42.0.X/X
'''

_Obs.: Replace X/X by the IP and range you found and it's corresponding range, if you didn't take the same aproach as we did and connected the Board to a router, the IP and range is usually 192.168.X.X/XX_

Find the right device and it's IP and access the SSH by running:
    
'''sh
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa root@10.42.0.X
'''

> [!NOTE]
> Because of the elder operating system Bitmain on the board, Linux has some troble connecting with the board.The flags '+diffie-hellman-group14-sha1' and '+ssh-rsa' are needed for a SSH connection.

After accessing the SSH, it will ask something and will await for a input, type 'yes'.
After typing, the command line will ask for a password. The default password fof the Bitmain facotry firmware is usuaaly 'admin', if it doesn't work, try typing 'root'.
    
Once averything go well, you should see the following text in you terminal.
    
'''sh
root@antMiner:~#
'''
    
- Raspbarry Pi's SSH
2. 
