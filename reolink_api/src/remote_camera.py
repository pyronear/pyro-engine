import logging
import re
from io import BytesIO
from urllib.parse import urlparse, urlunparse

import requests
from PIL import Image
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

# Setup logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RemoteCamera")


class RemoteCamera:
    def __init__(self, url: str, timeout: int = 5):
        """Simple remote camera that retrieves one snapshot as a Pillow image."""
        self.url = url
        self.timeout = timeout
        self.cam_type = "static"  # for compatibility with engine logic
        self.last_images: dict[int, Image.Image] = {}
        logger.debug(f"Initialized RemoteCamera with URL={self.url}, timeout={self.timeout}")

    def _strip_credentials(self, parsed):
        """Remove embedded credentials from the URL and return (clean_url, (user, pass))."""
        netloc = parsed.netloc
        if "@" in netloc:
            creds, hostport = netloc.rsplit("@", 1)
            if ":" in creds:
                user, password = creds.split(":", 1)
            else:
                user, password = creds, ""
            cleaned = parsed._replace(netloc=hostport)
            logger.debug(f"Found embedded credentials user={user}, hostport={hostport}")
            return urlunparse(cleaned), (user, password)
        logger.debug("No embedded credentials found")
        return urlunparse(parsed), None

    def _extract_query_credentials(self, url):
        """Extract ?usr= and ?pwd= credentials if present."""
        m_usr = re.search(r"[?&]usr=([^&]+)", url)
        m_pwd = re.search(r"[?&]pwd=([^&]+)", url)
        if m_usr and m_pwd:
            user, pwd = m_usr.group(1), m_pwd.group(1)
            logger.debug(f"Found query credentials user={user}")
            return (user, pwd)
        logger.debug("No query credentials found")
        return None

    def capture(self):
        """Capture one snapshot from the URL and return a Pillow Image (or None if failed)."""
        logger.info(f"Starting capture for {self.url}")
        headers = {"User-Agent": "python-requests/2.x", "Accept": "image/*"}
        parsed = urlparse(self.url)
        clean_url, auth_tuple = self._strip_credentials(parsed)
        query_auth = self._extract_query_credentials(self.url)

        # Prepare authentication candidates
        auth_candidates = []
        if query_auth:
            u, p = query_auth
            auth_candidates.append(HTTPBasicAuth(u, p))
            auth_candidates.append(HTTPDigestAuth(u, p))
        if auth_tuple:
            u, p = auth_tuple
            auth_candidates.append(HTTPBasicAuth(u, p))
            auth_candidates.append(HTTPDigestAuth(u, p))

        logger.debug(f"Prepared {len(auth_candidates)} authentication candidates")

        def fetch_image(target_url, auth=None):
            try:
                msg = f"Fetching image from {target_url}"
                if auth:
                    msg += f" using auth={auth.__class__.__name__}"
                logger.debug(msg)
                r = requests.get(target_url, headers=headers, timeout=self.timeout, auth=auth)
                logger.debug(
                    f"Response from {target_url}: status={r.status_code}, content_length={len(r.content) if r.content else 0}"
                )
                if r.status_code == 200 and r.content:
                    try:
                        img = Image.open(BytesIO(r.content)).convert("RGB")
                        logger.info(f"Image successfully loaded from {target_url}, size={img.size}")
                        return img
                    except Exception as e:
                        logger.error(f"Error decoding image from {target_url}: {e}")
            except requests.RequestException as e:
                logger.error(f"Request to {target_url} failed: {e}")
            return None

        # Try direct URL
        logger.debug(f"Trying direct URL: {self.url}")
        img = fetch_image(self.url)
        if img:
            logger.info(f"Capture successful from direct URL for {self.url}")
            return img

        # Try clean URL and auth combinations
        for auth in auth_candidates:
            for target in [self.url, clean_url]:
                logger.debug(f"Trying {target} with {auth.__class__.__name__}")
                img = fetch_image(target, auth)
                if img:
                    logger.info(f"Capture successful with {auth.__class__.__name__} for {target}")
                    return img

        logger.warning(f"All attempts failed for {self.url}")
        return None
