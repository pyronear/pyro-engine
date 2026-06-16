let PI_URL = "http://192.168.1.99:8081/health";

let OUTPUT_IDS = [0, 1];

let CHECK_INTERVAL_MS = 10 * 60 * 1000;
let REBOOT_DURATION_MS = 20 * 1000;

let MAX_FAILURES = 3;
let MAX_REBOOTS_PER_DAY = 3;

let failures = 0;
let rebooting = false;

let currentDay = "";
let rebootsToday = 0;

function getDayKey() {
  let now = new Date();
  return now.getFullYear() + "/" + (now.getMonth() + 1) + "/" + now.getDate();
}

function resetDailyCounterIfNeeded() {
  let dayKey = getDayKey();

  if (currentDay === "") {
    currentDay = dayKey;
    return;
  }

  if (dayKey !== currentDay) {
    currentDay = dayKey;
    rebootsToday = 0;
    print("New day, reboot counter reset");
  }
}

function canReboot() {
  resetDailyCounterIfNeeded();

  if (rebootsToday >= MAX_REBOOTS_PER_DAY) {
    print("Daily reboot limit reached, skip reboot");
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
  rebootsToday = rebootsToday + 1;

  print("Pi unreachable, rebooting outputs 0 and 1");
  print("Reboots today: " + rebootsToday);

  turnOutputs(false);

  Timer.set(REBOOT_DURATION_MS, false, function () {
    turnOutputs(true);
    rebooting = false;
    print("Outputs 0 and 1 are back on");
  });
}

function checkPi() {
  resetDailyCounterIfNeeded();

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

resetDailyCounterIfNeeded();
Timer.set(CHECK_INTERVAL_MS, true, checkPi);
checkPi();
