import json
import socket


with open('../../server/data/ip_hostname_mapping.json') as f:
    ip_hostname_mapping = json.load(f)

inv_map = {v: k for k, v in ip_hostname_mapping.items()}
hostnames = inv_map.keys()

ip_hostname_mapping = {}

for hostname in hostnames:
    try:
        ip = socket.gethostbyname(hostname + '.local')
        ip_hostname_mapping[ip] = hostname
    except Exception:
        ip = inv_map[hostname]
        ip_hostname_mapping[ip] = hostname


with open('../../server/data/ip_hostname_mapping.json', 'w') as fp:
    json.dump(ip_hostname_mapping, fp, indent=4)
