from fastapi import APIRouter

from camera.config import RAW_CONFIG

router = APIRouter()


@router.get("/cameras")
def list_cameras():
    return {"camera_ips": list(RAW_CONFIG.keys())}


@router.get("/camera_infos")
def get_camera_infos():
    """Returns list of cameras with their IP addresses, azimuths, and other metadata."""
    camera_infos = []

    for ip, conf in RAW_CONFIG.items():
        camera_infos.append({
            "ip": ip,
            "azimuths": conf.get("azimuths", []),
            "poses": conf.get("poses", []),
            "name": conf.get("name", "Unknown"),
            "id": conf.get("id"),
            "type": conf.get("type", "Unknown"),
            "brand": conf.get("brand", "unknown"),
        })

    return {"cameras": camera_infos}
