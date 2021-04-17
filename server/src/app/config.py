# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import json

PROJECT_NAME: str = 'Pyronear local webserver'
PROJECT_DESCRIPTION: str = 'API for pyronear systems interactions on sites'
API_BASE: str = 'api/'
VERSION: str = "0.1.0a0"
LOGO_URL: str = "https://github.com/pyronear/PyroNear/raw/master/docs/source/_static/img/pyronear-logo-dark.png"


# security
JWT_ENCODING_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 365 * 10

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"

# fleet db to be readed from env of json file outside the repo
with open('fleet_accesses.json') as f:
    fleet_accesses = json.load(f)
