"""Never raises. 'D:20230211100457+01'00'' -> '2023-02-11T10:04:57+01:00'."""
import re
from datetime import datetime, timedelta, timezone

_PDF_DATE = re.compile(
    r"""^\s*D?:?                                   # optional 'D:' prefix
        (\d{4})(\d{2})?(\d{2})?                   # year month day
        (\d{2})?(\d{2})?(\d{2})?                  # hour minute second
        (?:([Zz+\-])(\d{2})?'?(\d{2})?'?)?        # tz: Z | +hh'mm'
        \s*$""",
    re.VERBOSE,
)

def normalize_pdf_date(raw: str | None) -> str | None:
    if not raw:
        return None
    m = _PDF_DATE.match(str(raw))
    if not m:
        return None
    y, mo, d, h, mi, s, tzs, tzh, tzm = m.groups()
    try:
        dt = datetime(int(y), int(mo or 1), int(d or 1),
                      int(h or 0), int(mi or 0), int(s or 0))
    except ValueError:
        return None                       # e.g. month 13 — garbage in, None out
    if tzs in ("+", "-"):
        off = timedelta(hours=int(tzh or 0), minutes=int(tzm or 0))
        dt = dt.replace(tzinfo=timezone(-off if tzs == "-" else off))
    else:
        dt = dt.replace(tzinfo=timezone.utc)   # naive/Z -> assume UTC
    return dt.isoformat(timespec="seconds")