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

## Requirements

The machine running the setup script must be able to reach the Shelly on the local network.

Example:

```bash
curl "http://192.168.1.97/rpc/Shelly.GetDeviceInfo"
```

## First install

```bash
cd watchdog/shelly/device
chmod +x setup_shelly_watchdog.sh
./setup_shelly_watchdog.sh
```

## Update existing script

After editing `watchdog.js`:

```bash
./update_shelly_watchdog.sh
```

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
