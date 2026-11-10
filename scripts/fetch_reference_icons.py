"""Download open-licensed pictograms of airport facilities into airport-icons/reference/.

Sets (all free to reuse; licence recorded per file in airport-icons/reference/manifest.json):
  * maki        — Mapbox map POI icons, CC0 1.0
  * temaki      — Rapid editor's extension of Maki (indoor, transport), CC0 1.0
  * osm-carto   — OpenStreetMap's map symbols, CC0 1.0
  * material    — Google Material Symbols, Apache 2.0
  * iso7001     — ISO 7001 public information symbols redrawn on Wikimedia Commons, CC0
  * aiga        — AIGA / US DOT transportation symbols on Wikimedia Commons, public domain
Only icons whose name maps to a mapdiff category are kept (see CATEGORY_WORDS). SVGs are rendered locally with
PyMuPDF; nothing is executed from the downloads.

Run:  PYTHONNOUSERSITE=1 python scripts/fetch_reference_icons.py [--sets maki temaki ...]
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pymupdf
from PIL import Image

from mapdiff.icons import library_dir, to_rgb
from mapdiff.taxonomy import Category

AGENT = "mapdiff-reference-icons/0.1 (airport floorplan research)"
PX = 128
C = Category

# Words in an icon's name → category. More specific words first; the first match wins. A word matches as a whole
# word; a word ending in "*" also matches as the start of a longer word ("smok*" → smoking).
CATEGORY_WORDS: list[tuple[str, Category]] = [
    ("baggage_claim", C.BAGGAGE_CLAIM), ("baggage claim", C.BAGGAGE_CLAIM), ("baggage reclaim", C.BAGGAGE_CLAIM),
    ("arriving*", C.BAGGAGE_CLAIM), ("flight_land", C.BAGGAGE_CLAIM),
    ("baggage check", C.BAGGAGE_DROP), ("luggage*", C.BAGGAGE_SERVICE), ("baggage*", C.BAGGAGE_SERVICE),
    ("locker*", C.BAGGAGE_SERVICE), ("trolley*", C.BAGGAGE_SERVICE), ("lost", C.LOST_FOUND),
    ("passport*", C.PASSPORT_CONTROL), ("immigration*", C.PASSPORT_CONTROL), ("border", C.PASSPORT_CONTROL),
    ("customs*", C.CUSTOMS), ("security*", C.SECURITY_CHECK), ("check_in", C.CHECK_IN), ("check-in", C.CHECK_IN),
    ("check in", C.CHECK_IN), ("ticket*", C.TICKET_SALES), ("transfer*", C.TRANSFER_DESK), ("connecting*", C.TRANSFER_DESK),
    ("departing*", C.GATE), ("flight_takeoff", C.GATE), ("departure*", C.GATE), ("gate", C.GATE), ("airport", C.GATE),
    ("aircraft", C.GATE), ("airline_seat", C.LOUNGE), ("lounge*", C.LOUNGE), ("waiting*", C.SEATING_AREA),
    ("seat*", C.SEATING_AREA), ("bench*", C.SEATING_AREA),
    ("toilets_wheelchair", C.TOILET_ACCESSIBLE), ("accessible toilet", C.TOILET_ACCESSIBLE),
    ("toilet*", C.TOILET), ("restroom*", C.TOILET), ("lavator*", C.TOILET), ("wc", C.TOILET), ("man", C.TOILET),
    ("woman", C.TOILET), ("women", C.TOILET), ("men", C.TOILET), ("baby", C.BABY_CARE), ("nursery*", C.BABY_CARE),
    ("changing*", C.BABY_CARE), ("prayer*", C.PRAYER_ROOM), ("place_of_worship", C.PRAYER_ROOM), ("religious*", C.PRAYER_ROOM),
    ("chapel", C.PRAYER_ROOM), ("mosque", C.PRAYER_ROOM), ("smok*", C.SMOKING_AREA), ("drinking_water*", C.WATER_FOUNTAIN),
    ("drinking water*", C.WATER_FOUNTAIN), ("water_fountain", C.WATER_FOUNTAIN), ("fountain*", C.WATER_FOUNTAIN),
    ("telephone*", C.PHONE), ("phone", C.PHONE), ("post_box*", C.MAILBOX), ("post_office*", C.MAILBOX), ("mail", C.MAILBOX),
    ("letter", C.MAILBOX), ("charging*", C.CHARGING), ("battery", C.CHARGING), ("power", C.CHARGING),
    ("information", C.INFORMATION), ("info*", C.INFORMATION), ("help*", C.INFORMATION), ("meeting*", C.INFORMATION),
    ("flight*", C.FLIGHT_INFO), ("departure_board", C.FLIGHT_INFO), ("timetable", C.FLIGHT_INFO),
    ("wheelchair*", C.SPECIAL_ASSISTANCE), ("accessib*", C.SPECIAL_ASSISTANCE), ("disab*", C.SPECIAL_ASSISTANCE),
    ("assist*", C.SPECIAL_ASSISTANCE), ("first_aid*", C.MEDICAL), ("first aid*", C.MEDICAL), ("firstaid*", C.MEDICAL),
    ("doctor*", C.MEDICAL), ("hospital*", C.MEDICAL), ("medical*", C.MEDICAL), ("clinic*", C.MEDICAL),
    ("defibrillator*", C.AED), ("aed", C.AED), ("pharmac*", C.PHARMACY), ("chemist", C.PHARMACY), ("police*", C.POLICE),
    ("coffee*", C.CAFE), ("cafe", C.CAFE), ("restaurant*", C.RESTAURANT), ("fast_food", C.FOOD_DRINK),
    ("fastfood", C.FOOD_DRINK), ("bar", C.BAR), ("pub", C.BAR), ("beer", C.BAR), ("food", C.FOOD_DRINK),
    ("snack*", C.FOOD_DRINK), ("bakery*", C.FOOD_DRINK), ("ice_cream", C.FOOD_DRINK), ("duty_free", C.DUTY_FREE),
    ("duty free", C.DUTY_FREE), ("perfume*", C.DUTY_FREE), ("shop*", C.RETAIL), ("store*", C.RETAIL), ("shopping", C.RETAIL),
    ("gift*", C.RETAIL), ("newsagent*", C.RETAIL), ("newspaper*", C.RETAIL), ("book", C.RETAIL), ("clothes", C.RETAIL),
    ("convenience*", C.RETAIL), ("atm", C.ATM), ("cash*", C.ATM), ("currency*", C.CURRENCY_EXCHANGE),
    ("exchange*", C.CURRENCY_EXCHANGE), ("bureau_de_change", C.CURRENCY_EXCHANGE), ("bank*", C.BANK),
    ("hotel*", C.HOTEL), ("bed", C.HOTEL), ("car_rental", C.CAR_RENTAL), ("car rental", C.CAR_RENTAL),
    ("carrental", C.CAR_RENTAL), ("rental*", C.CAR_RENTAL), ("elevator*", C.LIFT), ("lift", C.LIFT),
    ("escalator*", C.ESCALATOR), ("stair*", C.STAIRS), ("steps", C.STAIRS), ("moving_walkway", C.TRAVELATOR),
    ("travelator", C.TRAVELATOR), ("walkway*", C.TRAVELATOR), ("emergency_exit", C.ENTRANCE_EXIT),
    ("exit*", C.ENTRANCE_EXIT), ("entrance*", C.ENTRANCE_EXIT), ("entry", C.ENTRANCE_EXIT), ("door", C.ENTRANCE_EXIT),
    ("parking_meter", C.PARKING_PAYMENT), ("parking_payment", C.PARKING_PAYMENT), ("parking*", C.PARKING),
    ("car_park", C.PARKING), ("garage*", C.PARKING), ("taxi*", C.TAXI), ("bus*", C.BUS), ("coach*", C.BUS),
    ("shuttle*", C.BUS), ("railway*", C.TRAIN), ("train*", C.TRAIN), ("rail", C.TRAIN), ("metro*", C.TRAIN),
    ("subway*", C.TRAIN), ("tram*", C.TRAIN), ("monorail", C.TRAIN), ("rideshare*", C.RIDESHARE), ("car_pooling", C.RIDESHARE),
    ("drop_off", C.PICKUP_DROPOFF), ("pickup", C.PICKUP_DROPOFF), ("pick_up", C.PICKUP_DROPOFF),
]

# Material Symbols are listed by name (the repository is too large to list).
MATERIAL = [
    "wc", "man", "woman", "family_restroom", "baby_changing_station", "elevator", "escalator", "stairs",
    "local_cafe", "coffee", "restaurant", "fastfood", "local_bar", "bakery_dining", "local_atm", "atm",
    "currency_exchange", "account_balance", "luggage", "no_luggage", "work", "local_taxi", "directions_bus",
    "airport_shuttle", "train", "subway", "tram", "local_parking", "car_rental", "smoking_rooms", "local_pharmacy",
    "medical_services", "local_hospital", "local_police", "info", "help", "flight_takeoff", "flight_land",
    "connecting_airports", "airline_seat_recline_normal", "accessible", "wheelchair_pickup", "shopping_bag",
    "local_mall", "storefront", "hotel", "local_post_office", "mail", "phone", "local_phone", "ev_station",
    "battery_charging_full", "water_drop", "local_drink", "meeting_room", "door_front", "exit_to_app", "login",
    "logout", "emergency", "health_and_safety", "security", "badge", "chair", "weekend", "airplane_ticket",
    "confirmation_number", "local_convenience_store", "book", "local_florist", "spa", "child_friendly",
    "child_care", "mosque", "church", "temple_buddhist", "synagogue", "travel", "local_airport", "departure_board",
    "directions_walk", "moving", "hail", "local_gas_station", "pedal_bike", "two_wheeler", "local_grocery_store",
]

SETS = {
    "maki": dict(repo="mapbox/maki", branch="main", folder="icons/", licence="CC0 1.0"),
    "temaki": dict(repo="rapideditor/temaki", branch="main", folder="icons/", licence="CC0 1.0"),
    "osm-carto": dict(repo="gravitystorm/openstreetmap-carto", branch="master", folder="symbols/", licence="CC0 1.0"),
}
COMMONS = {
    "iso7001": (["ISO 7001 - Public facilities", "ISO 7001 - Transportation facilities", "ISO 7001 - Commercial facilities",
                 "ISO 7001 - Accessibility", "ISO 7001 - Behaviour of the public"]),
    "aiga": ["AIGA symbol signs"],
}
COMMONS_LICENCES = re.compile(r"^(public domain|pd|cc0)", re.I)


def get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": AGENT}), timeout=30) as r:
                return r.read()
        except OSError:
            time.sleep(3 * 2 ** attempt)  # rate limits: back off and retry
    raise RuntimeError(f"download failed: {url}")


# Prohibition signs and look-alike names that are not airport facilities.
EXCLUDE = re.compile(r"\b(no|not|do not|forbidden|prohibited)\b|^no[-_ ]|barrier|power|blood|boat|bicycle|bike|garden|"
                     r"gondola|drag|chair lift|taxiing|camera|lift gate|water taxi|flush|paper|internet|bunk|funicular|"
                     r"goods|vegetarian|priority|bar lift|horizontal bar|ceremonial|city gate|movie|storage|"
                     r"benchmark|tree and|sign and|hanging rail", re.I)


def category_of(name: str) -> Category | None:
    low = name.lower().replace("-", "_")
    if EXCLUDE.search(name.replace("_", " ").replace("-", " ")):
        return None
    for word, cat in CATEGORY_WORDS:
        stem = word.endswith("*")
        w = re.escape(word.rstrip("*"))
        if re.search(rf"(^|[^a-z]){w}" + ("" if stem else r"($|[^a-z])"), low):
            return cat
    return None


def render_svg(svg: bytes, px: int = PX) -> pymupdf.Pixmap:
    with pymupdf.open(stream=svg, filetype="svg") as doc:
        page = doc[0]
        zoom = px / max(page.rect.width, page.rect.height)
        return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)


def github_icons(name: str, spec: dict) -> list[dict]:
    tree = json.loads(get(f"https://api.github.com/repos/{spec['repo']}/git/trees/{spec['branch']}?recursive=1"))
    paths = [t["path"] for t in tree["tree"] if t["path"].startswith(spec["folder"]) and t["path"].endswith(".svg")]
    out = []
    for path in paths:
        meaning = Path(path).stem.replace("-", " ").replace("_", " ").strip()
        out.append(dict(set=name, meaning=meaning, key=Path(path).stem, licence=spec["licence"],
                        url=f"https://raw.githubusercontent.com/{spec['repo']}/{spec['branch']}/{path}", kind="svg"))
    return out


def material_icons() -> list[dict]:
    base = "https://raw.githubusercontent.com/google/material-design-icons/master/symbols/web"
    return [dict(set="material", meaning=n.replace("_", " "), key=n, licence="Apache 2.0", kind="svg",
                 url=f"{base}/{n}/materialsymbolsoutlined/{n}_48px.svg") for n in MATERIAL]


def commons_icons(name: str, categories: list[str]) -> list[dict]:
    api = "https://commons.wikimedia.org/w/api.php?"
    titles: list[str] = []
    for cat in categories:
        d = json.loads(get(api + urlencode({"format": "json", "action": "query", "list": "categorymembers",
                                            "cmtitle": f"Category:{cat}", "cmlimit": 500, "cmtype": "file"})))
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        time.sleep(1)
    out = []
    titles = sorted(set(titles))
    for i in range(0, len(titles), 40):
        d = json.loads(get(api + urlencode({"format": "json", "action": "query", "prop": "imageinfo",
                                            "iiprop": "url|extmetadata", "iiurlwidth": PX * 2,
                                            "titles": "|".join(titles[i:i + 40])})))
        for p in d["query"]["pages"].values():
            info = (p.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            licence = meta.get("LicenseShortName", {}).get("value", "")
            if not info.get("thumburl") or not COMMONS_LICENCES.match(licence):
                continue
            desc = re.sub(r"\s+", " ", html.unescape(re.sub("<[^>]+>", " ", meta.get("ImageDescription", {}).get("value", ""))))
            stem = p["title"][5:].rsplit(".", 1)[0]
            m = re.match(r"\s*ISO 7001 [A-Z]{2} [A-Z]{2} \d{3}\s*(.*)", desc)
            meaning = (m.group(1).split(";")[0] if m else re.sub(r"^(aiga|\d+)[ _]*", "", stem, flags=re.I)).strip() or stem
            out.append(dict(set=name, meaning=meaning, key=stem, licence=licence, url=info["thumburl"], kind="png",
                            page=info.get("descriptionurl")))
        time.sleep(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(library_dir() / "reference"))
    ap.add_argument("--sets", nargs="+", default=[*SETS, "material", *COMMONS])
    args = ap.parse_args()
    out = Path(args.out)
    rejected_file = out / "rejected.json"  # icons a person removed: never downloaded again
    rejected = json.loads(rejected_file.read_text()) if rejected_file.exists() else []
    skip = {r["file"] for r in rejected}
    fixed = {r["file"]: r["category"] for r in (json.loads((out / "recategorised.json").read_text())
                                                 if (out / "recategorised.json").exists() else [])}
    manifest: list[dict] = []
    for name in args.sets:
        listing = (github_icons(name, SETS[name]) if name in SETS else material_icons() if name == "material"
                   else commons_icons(name, COMMONS[name]))
        kept = [(i, cat) for i in listing if (cat := category_of(f"{i['key']} {i['meaning']}"))]
        print(f"{name}: {len(listing)} icons, {len(kept)} airport-relevant")
        for icon, cat in kept:
            rel = f"{name}/{re.sub(r'[^a-z0-9]+', '-', icon['key'].lower()).strip('-')}.png"
            if rel in skip:
                continue
            cat = Category(fixed.get(rel, cat.value))
            target = out / rel
            if not target.exists():
                try:
                    data = get(icon["url"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if icon["kind"] == "svg":
                        render_svg(data).save(target)
                    else:
                        to_rgb(Image.open(io.BytesIO(data))).save(target)
                except (RuntimeError, ValueError, pymupdf.FileDataError) as e:
                    print(f"  skip {icon['key']}: {e}")
                    continue
                time.sleep(0.3)
            manifest.append(dict(file=rel, meaning=icon["meaning"], category=cat.value, set=name,
                                 licence=icon["licence"], source=icon.get("page") or icon["url"]))
    manifest.sort(key=lambda m: (m["category"], m["set"], m["file"]))
    keep = {m["file"] for m in manifest}
    for f in out.glob("*/*.png"):  # icons no longer selected
        if f"{f.parent.name}/{f.name}" not in keep:
            f.unlink()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"{len(manifest)} reference icons → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
