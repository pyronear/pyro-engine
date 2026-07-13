let PI_URL = "http://192.168.1.99:8081/health";

let OUTPUT_IDS = [0, 1];

let CHECK_INTERVAL_MS = 10 * 60 * 1000;
// A crashed Pi draws almost no current, so the PSU capacitors keep it powered
// through a short cut. 3 minutes guarantees a real power loss.
let REBOOT_DURATION_MS = 3 * 60 * 1000;

let MAX_FAILURES = 3;
let MAX_REBOOTS_PER_WINDOW = 3;
// The Shelly has no NTP when the station is offline, so wall-clock dates are
// unreliable. Timers only depend on uptime, so the budget is reset by a
// repeating 24h timer instead of a date change.
let RESET_WINDOW_MS = 24 * 60 * 60 * 1000;

let failures = 0;
let rebooting = false;
let rebootCount = 0;

function resetRebootBudget(reason) {
  if (rebootCount > 0) {
    print("Reboot budget reset (" + reason + ")");
  }
  rebootCount = 0;
}

function canReboot() {
  if (rebootCount >= MAX_REBOOTS_PER_WINDOW) {
    print("Reboot limit reached, skip reboot");
    return false;
  }

  return true;
}

function turnOutputs(on) {
  for (let i = 0; i < OUTPUT_IDS.length; i++) {
    Shelly.call("Switch.Set", { id: OUTPUT_IDS[i], on: on });
  }
}

function rebootOutputs() {
  if (rebooting) {
    print("Reboot already running, skip");
    return;
  }

  if (!canReboot()) {
    return;
  }

  rebooting = true;
  rebootCount = rebootCount + 1;

  print("Pi unreachable, rebooting outputs 0 and 1");
  print("Reboots this window: " + rebootCount);

  turnOutputs(false);

  Timer.set(REBOOT_DURATION_MS, false, function () {
    turnOutputs(true);
    rebooting = false;
    print("Outputs 0 and 1 are back on");
  });
}

function checkPi() {
  if (rebooting) {
    print("Reboot running, skip check");
    return;
  }

  Shelly.call(
    "HTTP.GET",
    { url: PI_URL, timeout: 5 },
    function (result, error_code, error_message) {
      if (error_code === 0 && result && result.code === 200) {
        failures = 0;
        resetRebootBudget("Pi healthy");
      } else {
        failures = failures + 1;
        print("Pi failed, count " + failures + " (" + (error_message || "bad status") + ")");

        if (failures >= MAX_FAILURES) {
          failures = 0;
          rebootOutputs();
        }
      }
    }
  );
}

Timer.set(RESET_WINDOW_MS, true, function () {
  resetRebootBudget("24h window elapsed");
});
Timer.set(CHECK_INTERVAL_MS, true, checkPi);
checkPi();
