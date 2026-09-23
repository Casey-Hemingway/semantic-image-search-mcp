"""Source class for each photo: who took it, and with what.

Casey's sourcing hierarchy (23 Sep 2026): programme photos should now come
from recent trips shot by working photographers (Blair Millar, Sam Tarling,
Alex Hillary) on proper cameras. Original archival film scans are "magical
when used sparingly". The problem tier is amateur point-and-shoot and
old-phone snaps from past trips. Round-1 pairwise labels backed this up:
27% of pro-body photos were judged unusable, against 53% of compact/phone
photos.

Classes:
    pro      named photographer folders, pro camera bodies, iPhone 13 and later
    archive  the historical photograph folders (film scans)
    amateur  any other camera that left EXIF (compacts, older phones)
    unknown  no camera EXIF and not in a historical folder (exports, scans
             filed elsewhere); ranked neutral, since we can't tell
"""

import re
from typing import Optional

PRO_PHOTOGRAPHER_FOLDERS = ("BLAIR MILLAR", "SAM TARLING", "ALEX HILLARY")

ARCHIVE_FOLDER_PREFIXES = (
    "07 Historical Photographs",
    "08 Historical Photographs",
    "12 Ed Hillary",
    "13 Hillary Family",
)

# Interchangeable-lens and pro bodies seen in the library, by EXIF model name.
_PRO_BODY = re.compile(
    r"EOS (5D|6D|7D|R\d?\b|[1-9]0D|M\d+)"
    r"|NIKON (D[5-9]\d\d|Z ?\d)"
    r"|\bX-(H\d|T\d+|Pro\d|S\d+)"
    r"|ILCE-|DSLR-A|\bSLT-"
    r"|DMC-G[X1-9H]|DC-G[H9X]",
    re.IGNORECASE,
)
_RECENT_IPHONE = re.compile(r"iPhone (1[3-9]|[2-9]\d)")

SOURCE_CLASSES = ("pro", "archive", "amateur", "unknown")


def classify_source(folder: Optional[str], camera_model: Optional[str]) -> str:
    folder = folder or ""
    camera = (camera_model or "").strip()
    if folder.startswith(ARCHIVE_FOLDER_PREFIXES):
        return "archive"
    upper = folder.upper()
    if any(name in upper for name in PRO_PHOTOGRAPHER_FOLDERS):
        return "pro"
    if camera and (_PRO_BODY.search(camera) or _RECENT_IPHONE.search(camera)):
        return "pro"
    if camera:
        return "amateur"
    return "unknown"
