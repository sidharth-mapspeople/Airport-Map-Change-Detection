"""Fixed, map-independent category taxonomy.

Every map uses its own words, colours and icons; the understanding agent maps
them onto these categories so that versions can be compared semantically.
Categories are grouped: a change *within* a group (e.g. cafe vs restaurant for
the same place) is treated as extraction noise, not a real change.
"""

from enum import StrEnum


class Category(StrEnum):
    # Amenities
    TOILET = "toilet"
    TOILET_ACCESSIBLE = "toilet_accessible"
    BABY_CARE = "baby_care"
    PRAYER_ROOM = "prayer_room"
    SMOKING_AREA = "smoking_area"
    SEATING_AREA = "seating_area"
    WATER_FOUNTAIN = "water_fountain"
    PHONE = "phone"
    MAILBOX = "mailbox"
    CHARGING = "charging"
    # Information & assistance
    INFORMATION = "information"
    FLIGHT_INFO = "flight_info"
    SPECIAL_ASSISTANCE = "special_assistance"
    LOST_FOUND = "lost_found"
    # Medical / safety
    PHARMACY = "pharmacy"
    MEDICAL = "medical"
    AED = "aed"
    POLICE = "police"
    # Processing
    CHECK_IN = "check_in"
    AIRLINE_DESK = "airline_desk"
    TICKET_SALES = "ticket_sales"
    BAGGAGE_DROP = "baggage_drop"
    BAGGAGE_CLAIM = "baggage_claim"
    BAGGAGE_SERVICE = "baggage_service"
    SECURITY_CHECK = "security_check"
    PASSPORT_CONTROL = "passport_control"
    CUSTOMS = "customs"
    TRANSFER_DESK = "transfer_desk"
    GATE = "gate"
    LOUNGE = "lounge"
    # Commercial
    CAFE = "cafe"
    RESTAURANT = "restaurant"
    BAR = "bar"
    FOOD_DRINK = "food_drink"
    RETAIL = "retail"
    DUTY_FREE = "duty_free"
    ATM = "atm"
    CURRENCY_EXCHANGE = "currency_exchange"
    BANK = "bank"
    SERVICE_COUNTER = "service_counter"
    HOTEL = "hotel"
    CAR_RENTAL = "car_rental"
    # Vertical circulation
    LIFT = "lift"
    ESCALATOR = "escalator"
    STAIRS = "stairs"
    TRAVELATOR = "travelator"
    # Access / transport
    ENTRANCE_EXIT = "entrance_exit"
    PARKING = "parking"
    PARKING_PAYMENT = "parking_payment"
    TAXI = "taxi"
    BUS = "bus"
    TRAIN = "train"
    RIDESHARE = "rideshare"
    PICKUP_DROPOFF = "pickup_dropoff"
    # Spatial
    ZONE = "zone"
    OTHER = "other"


C = Category
_GROUPS: dict[str, set[Category]] = {
    "toilets": {C.TOILET, C.TOILET_ACCESSIBLE},
    "food_drink": {C.CAFE, C.RESTAURANT, C.BAR, C.FOOD_DRINK},
    "retail": {C.RETAIL, C.DUTY_FREE, C.PHARMACY},
    "money": {C.ATM, C.CURRENCY_EXCHANGE, C.BANK},
    "information": {C.INFORMATION, C.FLIGHT_INFO, C.SPECIAL_ASSISTANCE},
    "medical": {C.MEDICAL, C.AED},
    "check_in": {C.CHECK_IN, C.AIRLINE_DESK, C.TICKET_SALES, C.BAGGAGE_DROP, C.TRANSFER_DESK},
    "border": {C.PASSPORT_CONTROL, C.CUSTOMS},
    "baggage": {C.BAGGAGE_CLAIM, C.BAGGAGE_SERVICE},
    "vertical": {C.LIFT, C.ESCALATOR, C.STAIRS, C.TRAVELATOR},
    "ground_transport": {C.TAXI, C.BUS, C.TRAIN, C.RIDESHARE, C.PICKUP_DROPOFF},
    "parking": {C.PARKING, C.PARKING_PAYMENT},
}
_GROUP_OF: dict[Category, str] = {cat: g for g, cats in _GROUPS.items() for cat in cats}


# Facilities that carry no specific name ("Toilets" is what it is, not who it is).
GENERIC = {C.TOILET, C.TOILET_ACCESSIBLE, C.LIFT, C.ESCALATOR, C.STAIRS, C.TRAVELATOR, C.ATM, C.WATER_FOUNTAIN,
           C.PHONE, C.MAILBOX, C.CHARGING, C.AED, C.SMOKING_AREA, C.SEATING_AREA, C.BABY_CARE}

# Relations worth comparing between versions: facts about who uses which facility. Anything else a model might
# record (map titles, legend contents, insets, marker numbers, car parks of a terminal) is presentation or
# duplicates what the entities already say, and only adds noise to the diff.
RELATION_PREDICATES: dict[str, str] = {
    "checks_in_at": "airline → check-in desk/island, e.g. 'Emirates' → 'Check-in 14'",
    "boards_at": "airline or flight group → gate/pier, e.g. 'Schengen flights' → 'Pier A'",
    "handled_by": "airline → ground handler or transfer desk, e.g. 'TAROM' → 'Aviapartner'",
}


def group_of(category: Category) -> str:
    """Coarse group used for change detection; ungrouped categories are their own group."""
    return _GROUP_OF.get(category, category.value)
