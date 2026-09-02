# Camera adapters

## What is an adapter for?

Every camera brand speaks its own protocol: HTTP CGI on Reolink, ISAPI on Linovision, raw RTSP,
or an in-house JSON API. An adapter translates that dialect into a small set of common methods,
so the REST API above it knows nothing about brands and the same routes work everywhere.

```
credentials.json          registry.py                adapters/              api/
  "adapter": "reolink" --> build_camera_object() --> ReolinkCamera --+
  "adapter": "rtsp"    -->   (dispatch on the    --> RTSPCamera -----+--> /capture
  "adapter": "rest"    -->    "adapter" field)   --> RestSnapshot ---+    /control/move
                                                                         /focus/...
```

Paths below are relative to the package root, `pyro_camera_api/pyro_camera_api/`: the contract is
`camera/base.py`, the dispatch `camera/registry.py`, the implementations `camera/adapters/` (this
directory).

## Is my camera already supported?

`rtsp`, `url` and `rest` are generic and cover many cameras with configuration alone.

| `adapter`                          | Class                | Capture | PTZ | Focus | Configuration you normally set                                        |
| ---------------------------------- | -------------------- | ------- | --- | ----- | --------------------------------------------------------------------- |
| `reolink-823S2`, `reolink-823A16`  | `ReolinkCamera`      | yes     | yes | yes   | `ip_address`, `poses`, `azimuths`                                     |
| `linovision` (alias `hikvision`)   | `LinovisionCamera`   | yes     | yes | yes   | `ip_address`, `poses`, `azimuths`                                     |
| `rtsp`                             | `RTSPCamera`         | yes     | no  | no    | `rtsp_url` (**required**)                                             |
| `url` (alias `http`, `https`)      | `URLCamera`          | yes     | no  | no    | `url` (**required**) with embedded credentials                        |
| `rest` (alias `api`)               | `RestSnapshotCamera` | yes     | no  | no    | `url` (**required**), `headers`, `response`, `json_path`, `encoding`  |
| `mock`                             | `MockCamera`         | yes     | yes | yes   | optional `url`, for tests and demos                                   |

Only the fields marked **required** block registration; the others have fallbacks (`ip_address`
defaults to the `credentials.json` key, `poses` and `azimuths` to empty lists), so a typo there
gives you a camera that registers and then misbehaves.

Picking between the generic three:

- RTSP stream: `rtsp`.
- Snapshot URL: `url`, but check the auth first. `URLCamera.capture()` branches on the literal
  string `CGIProxy.fcgi`: Foscam URLs carrying `?usr=...&pwd=...` are sent as is, everything else
  goes down the HTTP Digest branch and needs `user:pass@host` in the URL. A credential-free URL
  fails there rather than just working, use `rest` instead.
- HTTP endpoint with auth headers, or an image wrapped in JSON: `rest`. `json_path` and `encoding`
  are only read when `response` is `"json"`, so leaving `response` at its `"image"` default
  silently ignores them. Header and URL values accept `${VAR}` to keep secrets in `.env`.

Write a new adapter only for a proprietary control protocol, typically PTZ or focus, that none of
the three can drive.

## The capture() contract

One method is mandatory, whatever the camera:

```python
class BaseCamera(ABC):
    @abstractmethod
    def capture(self, **kwargs) -> Optional[Image.Image]: ...
```

- Return an RGB Pillow image, or `None` on failure. Never let an exception escape: the patrol and
  inference loops run continuously and read `None` as "no image this time".
- Log the failure before returning `None`, masking credentials and tokens.
- Set a timeout on every network call, and accept `**kwargs`, since routes sometimes pass
  `patrol_id` that static adapters ignore.

## Capabilities

Capabilities are mixins; routes check `isinstance()` and return 400 when the camera lacks them.

`PTZMixin`, pan/tilt/zoom:

| Member                 | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `move_camera()`        | Runs `"Left"`, `"Right"`, `"Up"`, `"Down"`, `"Stop"`, `"ToPos"`, ...    |
| `get_azimuth()`        | Current real-world azimuth in degrees `[0, 360)`, or `None` if unknown  |
| `cam_poses`            | The camera's local presets                                             |
| `cam_azimuths`         | Matching real-world azimuths, index-aligned with `cam_poses`            |
| `azimuth_source`       | `"tracked"` (dead-reckoned server-side) or `"hardware"` (read back)     |
| `preset_move_hold_s`   | Lock held after a fire-and-forget move; `0` when the call blocks        |

`FocusMixin`, manual focus: `set_manual_focus(position)` and `get_focus_level()`, returning
`{"focus": int | None, "zoom": int | None}`.

Beyond the mixins, some routes resolve methods with `hasattr()`. These are declared nowhere in
`base.py`, so this is the implicit half of the contract:

| Method              | Route or service using it                                          | Missing |
| ------------------- | ------------------------------------------------------------------ | ------- |
| `set_auto_focus()`  | `POST /focus/set_autofocus`                                        | 400     |
| `start_zoom_focus()`| `POST /control/zoom/{camera_ip}/{level}`, zoom reset after a stream | 400     |
| `get_ptz_preset()`  | `GET /control/preset/list`                                         | 400     |
| `set_ptz_preset()`  | `POST /control/preset/set`                                         | 400     |
| `reboot_camera()`   | `POST /control/reboot/{camera_ip}`, stuck-camera detector          | 501     |

The `focus_position` attribute works the same way: when set, the patrol loop calls
`set_manual_focus()` with it once per cycle, after the return to the first pose, and skips it
while a stream runs.

`focus_finder()` is the exception. `POST /focus/focus_finder` gates on `isinstance(cam, FocusMixin)`
and then calls it unguarded, so inheriting `FocusMixin` commits you to implementing it: without it
the route raises `AttributeError` and returns 500. It takes `save_images` and `should_abort`, polls
`should_abort()` before each focus move, and raises `FocusAbortedError` when it fires.

These methods have no reference signature, so copy `reolink.py` and `linovision.py` (both here).
Comparing them shows what is fixed (name, arguments, return type) and what is not:

```python
# reolink.py
def reboot_camera(self) -> bool:
    url = self._build_url("Reboot")
    response = requests.post(url, json=[{"cmd": "Reboot"}], verify=False)
    ...

# linovision.py
def reboot_camera(self) -> bool:
    resp = self._request("PUT", "/ISAPI/System/reboot")
    return self._handle_response(resp, "Reboot requested") is not None
```

`LinovisionCamera.set_auto_focus()` shows a third option: a stub that logs a warning, so the route
answers 200 doing nothing instead of a clean 400. Only worth it when a 400 would break an existing
caller.

### How much you actually need

A static camera needs `capture()` and nothing else. A PTZ camera needs `capture()` plus a
`move_camera()` handling `"ToPos"`, because the patrol loop does nothing more than
`move_camera("ToPos", idx=pose, speed=50)`, wait, `capture()`, for each pose in `cam_poses`.
`get_azimuth()` is abstract on `PTZMixin` so it must exist, but returning `None` is fine until you
need azimuth tracking.

Presets have to exist on the camera before `"ToPos"` means anything: create them by hand in the
vendor's interface, or implement `set_ptz_preset()` and `get_ptz_preset()` and drive them from the
API (see `setup_presets/`). The second is worth it as soon as you deploy more than one camera.

## Writing an adapter

**1.** Create the class in `camera/adapters/my_camera.py`:

```python
class MyCamera(BaseCamera, PTZMixin):  # drop PTZMixin for a static camera
    def __init__(self, camera_id, ip_address, username, password,
                 cam_type="static", cam_poses=None, cam_azimuths=None):
        super().__init__(camera_id=camera_id, cam_type=cam_type)
        self.ip_address = ip_address
        self.session = requests.Session()
        self.session.auth = (username, password)
        # Index-aligned: cam_poses[i] has real-world azimuth cam_azimuths[i]
        self.cam_poses = cam_poses or []
        self.cam_azimuths = cam_azimuths or []

    def capture(self, patrol_id=None, timeout=2):
        try:
            resp = self.session.get(f"http://{self.ip_address}/snapshot", timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            logger.error("Capture failed for %s: %s", self.camera_id, exc)
            return None  # never raise, the inference loop has to keep going

    def move_camera(self, operation, speed=20, idx=0):
        ...  # translate the operation into the camera's own protocol

    def get_azimuth(self):
        return self.current_azimuth  # None until a reference is known
```

**2.** Wire the dispatch in `build_camera_object()` (`camera/registry.py`), before the final error
branch. Return `None` with an error log when a required field is missing, so the camera is skipped
and the rest of the fleet still starts.

```python
if adapter == "my-camera":
    return MyCamera(
        camera_id=key,
        ip_address=ip_addr,
        username=CAM_USER or "",
        password=CAM_PWD or "",
        cam_type=cam_type,
        cam_poses=conf.get("poses", []),
        cam_azimuths=conf.get("azimuths", []),
    )
```

**3.** Export it in `camera/adapters/__init__.py`. Convention rather than wiring, since the
registry imports adapter modules directly and that file currently lists four of the six.

**4.** Add a test under `pyro_camera_api/tests/` building the camera with mocked HTTP and checking
`capture()`. `test_rest_snapshot.py` is the model.

## Pitfalls

- Reolink speed tables are keyed by exact model name, so a generic `adapter: "reolink"` builds the
  camera but falls back to the `823S2` calibration with only a warning, skewing degree-based moves.
- Streaming is outside the adapter scope: stream URLs are assembled in `core/config.py` from the
  IP, assuming RTSP on 554 with a Reolink or Linovision path. An adapter has nothing to implement
  and no say, so a camera off that convention captures fine and cannot stream. Moving URL
  construction into the adapters would be more consistent; it is not done today.
- `type: "ptz"` starts the patrol loop and the stuck detector at startup, so an incomplete PTZ
  adapter makes them fail continuously.
- `azimuth_source` changes what `get_azimuth()` means: `"tracked"` is dead-reckoned server-side and
  goes stale on any continuous rotation until the next preset, `"hardware"` is read from the camera.
- The registry is built at import time, and an exception in `__init__` is caught and logged, so a
  broken camera is simply absent. Check startup logs, not just `GET /cameras_list`.
