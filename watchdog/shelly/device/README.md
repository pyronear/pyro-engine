# Shelly Pi watchdog

This setup installs a Shelly script on a Shelly Pro device (works with any Pro
model that exposes the scripting RPC API, e.g. Pro 2PM, Pro 4PM).

It is the Shelly side of the Shelly-based watchdog. The Pi side lives in
`../main_pi/watchdog.py` and power-cycles output 0 when the internet or the
cameras are unreachable.

The Shelly checks the Raspberry Pi health endpoint every 10 minutes.

If the Pi health endpoint fails 3 times in a row, the Shelly reboots outputs 0 and 1.

Output mapping (RPC ids start at 0):

| Shelly output | RPC id |
| --- | --- |
| Output 1 | 0 |
| Output 2 | 1 |
| Output 3 | 2 |
| Output 4 | 3 |

Current configuration:

| Setting | Value |
| --- | --- |
| Pi health URL | `http://192.168.1.99:8081/health` |
| Check interval | 10 minutes |
| Consecutive failures before reboot | 3 |
| Reboot duration | 20 seconds |
| Outputs rebooted | 0 and 1 |
| Max reboots per day | 3 |

Important limitation:

The daily reboot counter is stored in script memory. If the Shelly itself reboots, the counter is reset.

Make sure the Shelly is not powered by one of the outputs it cuts.

## Setup procedure

### 1. Put the Shelly on the network

Power the Shelly, then use the **Shelly Smart Control** app (iOS / Android):

- The app connects to the device over **Bluetooth**.
- Add the device and configure it to join your local **Wi-Fi** (station mode).
- Give it a fixed address (static IP on the device, or a DHCP reservation on the
  router) so the watchdog can always reach it. The examples here assume
  `192.168.1.97`.

Check it is reachable from the machine that will run the scripts:

```bash
curl "http://192.168.1.97/rpc/Shelly.GetDeviceInfo"
```

### 2. Harden the Shelly

The Shelly is dedicated to running the watchdog, so disable everything else
with `harden_shelly.sh` (Cloud, MQTT, outbound WebSocket, Wi-Fi access point,
BLE, and any webhooks/schedules), enable eco mode to reduce heat, and force the
watchdog outputs (0 and 1) to power on at boot so a Shelly restart does not
leave the Pi unpowered:

```bash
cd watchdog/shelly/device
chmod +x harden_shelly.sh setup_shelly_watchdog.sh update_shelly_watchdog.sh
SHELLY_IP="192.168.1.97" ./harden_shelly.sh
```

After this the Shelly is only reachable over your main Wi-Fi (station mode), so
make sure step 1 worked first. If that link is later lost you need the physical
reset button to bring the access point back.

### 3. Install the Shelly watchdog script

Upload and start `watchdog.js` on the Shelly:

```bash
SHELLY_IP="192.168.1.97" ./setup_shelly_watchdog.sh
```

This is the Shelly side: it polls the Pi `/health` endpoint and reboots outputs
0 and 1 if the Pi is unhealthy. To change the target URL or timings, edit
`watchdog.js` and re-run:

```bash
SHELLY_IP="192.168.1.97" ./update_shelly_watchdog.sh
```

### 4. Install the Pi-side watchdog

The Pi side (`../main_pi/watchdog.py`) covers the other direction: it checks
internet connectivity and the cameras, and asks the Shelly to power-cycle
output 0 when they fail. Install it on the main Pi.

Create `/home/pi/watchdog.env` (optional — values below are the defaults):

```bash
SHELLY_IP=192.168.1.97
SHELLY_OUTPUT_ID=0
CAM_IPS=192.168.1.11,192.168.1.12
```

Add a cron entry (`crontab -e`) to run it every 10 minutes:

```cron
5,15,25,35,45,55 * * * * /usr/bin/python3 /home/pi/pyro-engine/watchdog/shelly/main_pi/watchdog.py >> /home/pi/watchdog_main.log 2>&1
```

Adjust the path to match where this repo lives on the Pi. It logs to
`/home/pi/watchdog_main.log`.

## Check status

```bash
curl "http://192.168.1.97/rpc/Script.List"
# use the id returned by Script.List (often 1 if it is the only script)
curl "http://192.168.1.97/rpc/Script.GetStatus?id=1"
```

Expected result:

```json
{"scripts":[{"id":1,"name":"pi_watchdog","enable":true,"running":true}]}
```
