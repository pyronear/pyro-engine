# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.api.schemas import Device, DeviceInDB, TokenData
from app import config as cfg


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login/access-token")


def get_device(db, username: str):
    if username in db:
        device_dict = db[username]
        return DeviceInDB(**device_dict)


async def get_current_device(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, cfg.SECRET_KEY, algorithms=[cfg.JWT_ENCODING_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    device = get_device(cfg.fleet_accesses, username=token_data.username)
    if device is None:
        raise credentials_exception
    return device


async def get_current_active_device(current_device: Device = Depends(get_current_device)):
    if current_device.disabled:
        raise HTTPException(status_code=400, detail="Inactive device")
    return current_device
