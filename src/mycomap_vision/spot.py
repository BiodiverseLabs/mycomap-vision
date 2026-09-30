"""The Spot interruption notice, read from the instance metadata service (IMDSv2).

AWS may take a Spot instance back at two minutes' notice. The notice appears at
http://169.254.169.254/latest/meta-data/spot/instance-action (404 until then) as
{"action": "terminate", "time": "..."}. `SpotWatcher` asks every few seconds in a
background thread, and once the notice is there it answers True when the trainer
asks whether to stop (trainer.run_job's `interrupted`). The job then abandons the
stage under way, marks the run "interrupted" and exits, well inside the two minutes.

Only the metadata service on the instance itself is asked; nothing leaves the box.
A failed or slow answer is never taken as a notice.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

IMDS = "http://169.254.169.254"
TOKEN_PATH = "/latest/api/token"
ACTION_PATH = "/latest/meta-data/spot/instance-action"
POLL_SECONDS = 5.0
TOKEN_TTL = 21600               # 6 h, the most IMDSv2 allows
TOKEN_REFRESH = 5 * 3600        # asked for again before it runs out
NOTICE_ACTIONS = ("terminate", "stop", "hibernate")


class SpotWatcher:
    """should_stop() for a Spot instance: True once AWS has given notice."""

    def __init__(self, session=None, interval: float = POLL_SECONDS, log=print,
                 clock: Callable[[], float] = time.monotonic):
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False           # never through a proxy: it is the local IMDS
        self.session, self.interval, self.log, self.clock = session, interval, log, clock
        self.notice: dict | None = None
        self._token: str | None = None
        self._token_at = 0.0
        self._done = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self) -> bool:
        return self.notice is not None

    def _get_token(self) -> str | None:
        if self._token and self.clock() - self._token_at < TOKEN_REFRESH:
            return self._token
        r = self.session.put(IMDS + TOKEN_PATH, timeout=2,
                             headers={"X-aws-ec2-metadata-token-ttl-seconds": str(TOKEN_TTL)})
        if r.status_code != 200:
            return None
        self._token, self._token_at = r.text.strip(), self.clock()
        return self._token

    def check(self) -> dict | None:
        """Ask once. The notice when there is one, else None; never raises."""
        if self.notice is not None:
            return self.notice
        try:
            token = self._get_token()
            if not token:
                return None
            r = self.session.get(IMDS + ACTION_PATH, timeout=2,
                                 headers={"X-aws-ec2-metadata-token": token})
            if r.status_code == 401:            # the token ran out: a new one next time
                self._token = None
                return None
            if r.status_code != 200:            # 404: no notice
                return None
            body = r.json()
        except Exception:                       # the service is slow or away: not a notice
            return None
        if isinstance(body, dict) and body.get("action") in NOTICE_ACTIONS:
            self.notice = body
            self.log(f"SPOT INTERRUPTION NOTICE: {body.get('action')} at {body.get('time')}; "
                     "stopping the stage under way")
            return body
        return None

    def start(self) -> "SpotWatcher":
        def loop():
            while not self._done.is_set():
                if self.check() is not None:
                    return
                self._done.wait(self.interval)
        self._thread = threading.Thread(target=loop, name="spot-watcher", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._done.set()
