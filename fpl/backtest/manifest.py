"""Append-only record of every forecast version, and which one was acted on.

The ledger kept immutable copies under `versions/` but nothing said which of
them mattered. `gw{n}.parquet` was overwritten on every ordinary run, and
`load_predictions` always read that mutable file, so a post-deadline re-run --
or a replay -- could silently replace the forecast the optimizer actually acted
on. Later calibration then fitted on the replacement, which is feedback
leakage: the model is being corrected against a forecast that was itself built
with knowledge the live run did not have.

Three rules follow, and they are the whole module:

* the manifest is APPEND-ONLY, because a later run must not be able to rewrite
  what an earlier one recorded;
* a version carries its ORIGIN, and a replay is never selectable as the record
  of a live gameweek;
* an ACTIONED marker beats recency, because a forecast is scored for having
  been acted on, not for having been last.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

LEDGER_DIR = "predictions"
MANIFEST_FILE = "manifest.jsonl"
# What a forecast was made for. "live" is a real pre-deadline run; "replay" is
# reconstructed after the fact from data the live model never had.
LIVE, REPLAY = "live", "replay"
# Last resort when the source tree is not a git checkout (an installed copy, a
# zip). Prefer the SHA: a hand-maintained date drifts behind the code.
FALLBACK_MODEL_VERSION = "unversioned"


def _instant(value) -> datetime | None:
    """An ISO timestamp as a comparable instant, or None.

    `created_at` is written as ISO with `+00:00`; FPL deadlines end in `Z`.
    Comparing those as strings happens to work for different dates and fails
    at the edges, so both are parsed. A naive value is taken as UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _before(created, deadline) -> bool:
    a, b = _instant(created), _instant(deadline)
    if a is None or b is None:
        return False
    return a < b


def _path(root) -> Path:
    return Path(root) / LEDGER_DIR / MANIFEST_FILE


def _append(root, record: dict) -> dict:
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _read(root) -> list[dict]:
    p = _path(root)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A half-written line from an interrupted run must not make the
            # whole ledger unreadable; the rest of the history is still good.
            continue
    return out


def model_version() -> str:
    """The commit this forecast was produced by, plus a dirty flag.

    `MODEL_VERSION` was a date maintained by hand and had fallen behind several
    core commits, so scored gameweeks were attributed to a model that was not
    the one that produced them.
    """
    try:
        here = Path(__file__).resolve().parent
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=here, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        if not sha:
            return FALLBACK_MODEL_VERSION
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=here, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return FALLBACK_MODEL_VERSION


def record_version(root, *, gw: int, version: str, created_at, origin: str = LIVE,
                   model_version: str = "", config_hash: str = "",
                   deadline: str | None = None,
                   snapshot: str | None = None) -> dict:
    """Append one forecast version. Never replaces an existing record.

    `snapshot` is the `fpl.data.snapshots` version this forecast read, so a
    replay of the acted-on forecast can reproduce exactly its inputs rather
    than whichever capture happens to be newest.
    """
    when = created_at or datetime.now(timezone.utc)
    return _append(root, {
        "kind": "version",
        "gw": int(gw),
        "version": str(version),
        "created_at": when.isoformat() if hasattr(when, "isoformat") else str(when),
        "origin": str(origin),
        "model_version": str(model_version),
        "config_hash": str(config_hash),
        "deadline": deadline,
        "snapshot": snapshot,
    })


def entries(root, gw: int | None = None) -> list[dict]:
    """Every recorded VERSION, oldest first. Actioned markers are not versions."""
    rows = [r for r in _read(root) if r.get("kind") == "version"]
    if gw is not None:
        rows = [r for r in rows if int(r.get("gw", -1)) == int(gw)]
    return rows


def _actioned(root, gw: int) -> str | None:
    """The version most recently marked as acted on for `gw`, if any."""
    marks = [r for r in _read(root)
             if r.get("kind") == "actioned" and int(r.get("gw", -1)) == int(gw)]
    return marks[-1]["version"] if marks else None


def mark_actioned(root, gw: int, version: str | None = None, when=None,
                  deadline: str | None = None) -> dict | None:
    """Record that a forecast version was the one acted on.

    Called when a gameweek is confirmed. Which version that is turns out to be
    the whole question, because a confirmation run writes its OWN forecast
    first: "the newest" is then the confirmation's forecast, not the planning
    one the manager looked at when deciding. So with a `deadline` the default
    is the newest LIVE version made strictly before it -- the last forecast the
    manager could actually have seen -- and a version made after the deadline
    is refused outright, explicitly named or not, because an actioned marker
    beats everything in `select_version` and a post-deadline forecast would
    carry the team news into calibration.

    Returns the version record that was marked, or None if nothing qualified.
    """
    live = [e for e in entries(root, gw) if e.get("origin") == LIVE]
    if not live:
        return None
    cutoff = deadline or next((e.get("deadline") for e in reversed(live)
                               if e.get("deadline")), None)

    if version is not None:
        chosen = next((e for e in reversed(live) if e["version"] == version), None)
        if chosen is None:
            return None
        if cutoff and not _before(chosen.get("created_at"), cutoff):
            return None
    elif cutoff:
        eligible = [e for e in live if _before(e.get("created_at"), cutoff)]
        if not eligible:
            return None
        chosen = eligible[-1]
    else:
        chosen = live[-1]

    stamp = when or datetime.now(timezone.utc)
    _append(root, {
        "kind": "actioned",
        "gw": int(gw),
        "version": chosen["version"],
        "actioned_at": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
    })
    return chosen


def select_version(root, gw: int, deadline: str | None = None) -> dict | None:
    """The forecast that should be scored and calibrated on for `gw`.

    In order of authority: the version explicitly marked as acted on; failing
    that the newest live version made strictly BEFORE the deadline, which is
    the last forecast the manager could have seen; failing that the newest live
    version at all, for a gameweek whose deadline was never recorded.

    A replay is never returned. It is built from today's prices, status and
    news rather than the deadline's, so scoring it measures a model that had
    information the live one did not.
    """
    live = [e for e in entries(root, gw) if e.get("origin") == LIVE]
    if not live:
        return None

    actioned = _actioned(root, gw)
    if actioned is not None:
        hit = next((e for e in reversed(live) if e["version"] == actioned), None)
        if hit is not None:
            return hit

    cutoff = deadline or next((e.get("deadline") for e in reversed(live)
                               if e.get("deadline")), None)
    if cutoff:
        before = [e for e in live if _before(e.get("created_at"), cutoff)]
        if before:
            return before[-1]
    return live[-1]
