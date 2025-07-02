import time

from fastapi import APIRouter, HTTPException

from camera.registry import CAMERA_REGISTRY
from camera.time_utils import update_command_time

router = APIRouter()


@router.post("/manual")
def manual_focus(camera_ip: str, position: int):
    update_command_time()
    cam = CAMERA_REGISTRY[camera_ip]
    print(cam.cam_type)
    result = cam.set_manual_focus(position)
    return {
        "status": "manual_focus",
        "camera_ip": camera_ip,
        "position": position,
        "result": result,
    }


@router.post("/set_autofocus")
def toggle_autofocus(camera_ip: str, disable: bool = True):
    update_command_time()
    cam = CAMERA_REGISTRY[camera_ip]
    result = cam.set_auto_focus(disable)
    return {
        "status": "autofocus",
        "camera_ip": camera_ip,
        "disabled": disable,
        "result": result,
    }


@router.get("/status")
def get_focus_status(camera_ip: str):
    update_command_time()
    cam = CAMERA_REGISTRY[camera_ip]
    data = cam.get_focus_level()
    if not data:
        raise HTTPException(status_code=500, detail="Could not retrieve focus level")
    return {"camera_ip": camera_ip, "focus_data": data}


@router.post("/focus_finder")
def run_focus_optimization(camera_ip: str, save_images: bool = False):
    cam = CAMERA_REGISTRY[camera_ip]

    if cam.cam_type == "static":
        raise HTTPException(status_code=400, detail="Autofocus is not supported for static cameras")

    # Optional move to pose 1
    if cam.cam_poses and len(cam.cam_poses) > 1:
        pose1 = cam.cam_poses[1]
        cam.move_camera("ToPos", idx=pose1, speed=50)
        time.sleep(1)

    # Run autofocus
    best_position = cam.focus_finder(save_images=save_images)

    return {"camera_ip": camera_ip, "best_focus_position": best_position, "status": "focus_updated"}
