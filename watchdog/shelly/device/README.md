# Shelly Pi watchdog

`watchdog.js` runs on a Shelly Pro device (works with any Pro model that
exposes the scripting RPC API, e.g. Pro 2PM, Pro 4PM).

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
| Reboot duration | 2 minutes |
| Outputs rebooted | 0 and 1 |
| Max reboots per outage | 3 |

The reboot duration is long on purpose: a crashed Pi draws almost no current,
so the power supply capacitors can keep it alive through a short cut.

The reboot budget (3 attempts) resets when the Pi answers a health check again,
or after 12 hours measured by a repeating timer. Timers only depend on uptime,
so the reset works even when the Shelly has no NTP/internet and its wall-clock
date is frozen.

Important limitation:

The reboot counter is stored in script memory. If the Shelly itself reboots, the counter is reset.

Make sure the Shelly is not powered by one of the outputs it cuts.

## Setup procedure

### 1. Set up the Shelly

Provisioning (Wi-Fi, static IP, firmware, hardening) and uploading
`watchdog.js` are done from
[pyro-sys-setup](https://github.com/pyronear/pyro-sys-setup#shelly-setup),
which reads `watchdog.js` from a pyro-engine checkout. After changing
`watchdog.js`, upload it again with `setup_shelly_watchdog.sh` from there.

### 2. Install the Pi-side watchdog

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
