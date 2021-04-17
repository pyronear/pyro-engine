from jose import jwt
from passlib.context import CryptContext
from typing import Optional
from datetime import datetime, timedelta

from app import config as cfg


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):

    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):

    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta):

    to_encode = data.copy()

    expire = datetime.utcnow() + expires_delta

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, cfg.SECRET_KEY, algorithm=cfg.JWT_ENCODING_ALGORITHM)

    return encoded_jwt
