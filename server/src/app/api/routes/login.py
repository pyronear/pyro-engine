# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import timedelta

from app.api.schemas import Device, DeviceInDB, Token, TokenData
from app.api import security
from app.api.deps import get_device

from app import config as cfg

router = APIRouter()


def authenticate_device(fake_db, username: str, password: str):

    device = get_device(fake_db, username)

    if not device:

        return False

    if not security.verify_password(password, device.hashed_password):

        return False

    return device


@router.post("/access-token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    This API follows the OAuth 2.0 specification

    If the credentials are valid, creates a new access token

    """

    # Verify device password
    device = authenticate_device(cfg.fleet_accesses, form_data.username, form_data.password)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # create access token
    access_token_expires = timedelta(minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": device.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}
