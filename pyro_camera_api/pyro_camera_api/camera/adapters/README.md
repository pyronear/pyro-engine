# Camera adapters

## What is an adapter for?

Every camera brand speaks its own protocol: HTTP CGI on Reolink, ISAPI on Linovision, raw RTSP,
or some in-house JSON API. An adapter translates that dialect into a small set of common
methods. Above it, the REST API knows nothing about brands, and the same routes work for every
camera.

```
credentials.json          registry.py                adapters/              api/
  "adapter": "reolink" --> build_camera_object() --> ReolinkCamera --+
  "adapter": "rtsp"    -->   (dispatch on the    --> RTSPCamera -----+--> /capture
  "adapter": "rest"    -->    "adapter" field)   --> RestSnapshot ---+    /control/move
                                                                         /focus/...
```

Paths below are given from the package root, `pyro_camera_api/pyro_camera_api/`. The contract
lives in `camera/base.py`, the dispatch in `camera/registry.py`, and the implementations in
`camera/adapters/` (this directory).

## Is my camera already supported?

Check this table before writing any code. The `rtsp`, `url` and `rest` adapters are generic and
cover many cameras without a single line of Python.

| `adapter`                          | Class                | Capture | PTZ | Focus | Required configuration                      |
| ---------------------------------- | -------------------- | ------- | --- | ----- | ------------------------------------------- |
| `reolink-823S2`, `reolink-823A16`  | `ReolinkCamera`      | yes     | yes | yes   | `ip_address`, `poses`, `azimuths`           |
| `linovision` (alias `hikvision`)   | `LinovisionCamera`   | yes     | yes | yes   | `ip_address`, `poses`, `azimuths`           |
| `rtsp`                             | `RTSPCamera`         | yes     | no  | no    | `rtsp_url`                                  |
| `url` (alias `http`, `https`)      | `URLCamera`          | yes     | no  | no    | `url` with embedded credentials             |
| `rest` (alias `api`)               | `RestSnapshotCamera` | yes     | no  | no    | `url`, `headers`, `json_path`, `encoding`   |
| `mock`                             | `MockCamera`         | yes     | yes | yes   | optional `url`, for tests and demos         |

Pick the generic adapter that matches:

- The camera exposes an RTSP stream, use `rtsp`.
- It exposes a snapshot URL returning an image directly, use `url`. Note that this adapter
  expects credentials inside the URL, either as Foscam-style query parameters
  (`?usr=...&pwd=...`) or as `user:pass@host`, in which case it switches to HTTP Digest
  authentication.
- It exposes an HTTP endpoint that needs authentication headers, or that wraps the image in
  JSON (base64 or a nested URL), use `rest`. Header and URL values accept `${VAR}` to read an
  environment variable, which keeps secrets out of `credentials.json`.

A new adapter is only justified when the camera has a proprietary control protocol, typically
PTZ or focus, which the three generic adapters cannot drive.

## What every adapter must implement

Exactly one method, whatever the camera:

```python
class BaseCamera(ABC):
    @abstractmethod
    def capture(self, **kwargs) -> Optional[Image.Image]: ...
```

Three rules:

- Return an RGB Pillow image, or `None` on failure. Never let an exception escape, because the
  patrol and inference loops run continuously and treat `None` as "no image this time".
- Log the failure before returning `None`, masking credentials and tokens.
- Always set a timeout on network calls, and accept `**kwargs`, since routes sometimes pass
  `patrol_id`, which static adapters ignore.

That is the whole mandatory surface: a static camera implementing only `capture()` is complete
and needs nothing below. Everything that follows is opt-in, and how far to go is answered in
[How far to go](#how-far-to-go).

## Adding capabilities

Capabilities are mixins. Routes check `isinstance()` and return an explicit 400 when the camera
does not implement them, so there is nothing to declare anywhere else.

**`PTZMixin`**, pan/tilt/zoom control:

| Member                 | Purpose                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `move_camera()`        | Runs `"Left"`, `"Right"`, `"Up"`, `"Down"`, `"Stop"`, `"ToPos"`, ...    |
| `get_azimuth()`        | Current real-world azimuth in degrees `[0, 360)`, or `None` if unknown  |
| `cam_poses`            | The camera's local presets                                             |
| `cam_azimuths`         | Matching real-world azimuths, index-aligned with `cam_poses`            |
| `azimuth_source`       | `"tracked"` (dead-reckoned server-side) or `"hardware"` (read back)     |
| `preset_move_hold_s`   | Lock held after a fire-and-forget move; `0` when the call blocks        |

**`FocusMixin`**, manual focus: `set_manual_focus(position)` and `get_focus_level()`, which
returns `{"focus": int | None, "zoom": int | None}`.

**Optional methods.** Beyond the mixins, some routes look methods up with `hasattr()` and
degrade gracefully when they are missing. These are declared nowhere in `base.py`, so this is
the implicit contract you need to know about:

| Method              | Route or service using it                             |
| ------------------- | ----------------------------------------------------- |
| `set_auto_focus()`  | `POST /focus/set_autofocus`                           |
| `focus_finder()`    | `POST /focus/focus_finder` (sharpness search)         |
| `start_zoom_focus()`| `POST /control/zoom`, zoom reset when a stream ends   |
| `get_ptz_preset()`  | `GET /control/preset/list`                            |
| `set_ptz_preset()`  | `POST /control/preset/set`                            |
| `reboot_camera()`   | `POST /control/reboot`, stuck-camera detector         |

One optional attribute plays the same role: when `focus_position` is set, the patrol loop calls
`set_manual_focus()` with that value on every pass over a preset.

Implement these case by case, only for the routes you actually need. A PTZ camera without
`reboot_camera()` works fine, it is simply left out of the stuck detector at startup.

Since these methods have no reference signature, **use `reolink.py` and `linovision.py` as your
model** (both in this directory). They are the two complete implementations, and comparing them
shows what is imposed
(the name, the arguments, the return type) and what is free (the entire protocol).
`reboot_camera()`, for instance, on Reolink:

```python
def reboot_camera(self) -> bool:
    url = self._build_url("Reboot")
    response = requests.post(url, json=[{"cmd": "Reboot"}], verify=False)
    ...
```

and on Linovision:

```python
def reboot_camera(self) -> bool:
    resp = self._request("PUT", "/ISAPI/System/reboot")
    return self._handle_response(resp, "Reboot requested") is not None
```

Same name, same `-> bool`, two unrelated protocols. Your adapter does the same with its own.

A third option, shown by `LinovisionCamera.set_auto_focus()`, is to declare the method but leave
it as a stub that logs a warning. The route then answers 200 while doing nothing, instead of the
clear 400 you get by not declaring the method at all. Keep this for cases where a 400 would
break an existing caller; otherwise, leaving the method out is the better answer.

### How far to go

Nothing above is required beyond `capture()`, so the real question is how much of it earns its
place. Two targets are enough to make a camera usable by the detection pipeline, and everything
else can be added later, on demand.

**Static camera: `capture()` only.** The inference loop runs, images flow, no mixin needed.

**PTZ camera: `capture()` plus a `move_camera()` handling at least `"ToPos"`.** The patrol loop
does nothing more than this: for each pose in `cam_poses` it calls
`move_camera("ToPos", idx=pose, speed=50)`, waits, then calls `capture()`. With those two
methods, patrol and multi-pose detection work end to end.

That leaves presets, which must exist on the camera before `"ToPos"` means anything. Two paths:
create them by hand in the vendor's interface, or implement `set_ptz_preset()` and
`get_ptz_preset()` to drive them from the API (`POST /control/preset/set`, see also the scripts
in `setup_presets/`). The second is strongly recommended as soon as several cameras are to be
deployed, since it is the difference between a reproducible install and a configuration clicked
site by site.

`get_azimuth()` is required by `PTZMixin` and must therefore exist, but it can simply return
`None` for as long as azimuth tracking is not needed.

## Writing an adapter

**1. Create the class** in `camera/adapters/my_camera.py` (this directory):

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

**2. Wire the dispatch** in `build_camera_object()` (`camera/registry.py`), before the final
error branch:

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

Return `None` with an error log when a required field is missing: the camera is skipped and the
rest of the fleet still starts.

**3. Export** the class in `camera/adapters/__init__.py`.

**4. Test** by adding a test under `pyro_camera_api/tests/` that builds the camera with mocked
HTTP requests and checks `capture()`. See `test_rest_snapshot.py` as a model.

## Pitfalls

- **Reolink speed tables are keyed by exact model name.** `"reolink-823S2"` and
  `"reolink-823A16"` have different calibrations. A generic `adapter: "reolink"` still builds
  the camera, but the PTZ routes fall back to `"reolink-823S2"` with a warning, which silently
  skews degree-based moves.
- **Streaming is outside the adapter scope, at this stage.** It was built separately: stream
  URLs are assembled in `core/config.py` from the IP, assuming RTSP on port 554 with a Reolink
  path by default, or a Linovision one. An adapter therefore has nothing to implement for
  streaming, but it also has no say in it, and a camera that does not follow that convention
  captures correctly yet cannot stream. Moving URL construction down into the adapters would be
  consistent with the rest of the design; it is not done today, and it is worth keeping in mind
  before integrating a camera whose stream does not look like `rtsp://<ip>:554/...`.
- **`type: "ptz"` starts background tasks.** At startup, the patrol loop and the stuck detector
  are launched for every camera marked `ptz`. An incomplete PTZ adapter will make those loops
  fail continuously.
- **`azimuth_source` changes what `get_azimuth()` means.** Under `"tracked"`, the azimuth is
  dead-reckoned server-side from commanded moves, and any continuous rotation makes it stale
  until the next preset. Under `"hardware"`, it is read from the camera and always trustworthy.
- **The registry is built at import time.** An exception in `__init__` is caught and logged, and
  the camera is simply absent from the registry. Check the startup logs, not just
  `GET /cameras_list`.
