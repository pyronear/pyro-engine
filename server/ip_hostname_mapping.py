import json
import socket


with open('data/ip_hostname_mapping.json') as f:
    ip_hostname_mapping = json.load(f)

inv_map = {v: k for k, v in ip_hostname_mapping.items()}
ip_hostname_mapping = {}
hostnames = inv_map.keys()
for hostname in hostnames:
    try:
        ip = socket.gethostbyname(hostname + '.local')
        ip_hostname_mapping[ip] = hostname
    except Exception:
        ip = inv_map[hostname]
        ip_hostname_mapping[ip] = hostname


with open('data/ip_hostname_mapping.json', 'w') as fp:
    json.dump(ip_hostname_mapping, fp, indent=4)
