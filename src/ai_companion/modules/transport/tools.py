from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain_core.tools import tool

from ai_companion.modules.calendar import _parse_local
from ai_companion.settings import settings


API_URL = "https://api.transitous.org/api/v1"
# Transitous is a free community service: identify yourself so they can contact you.
HEADERS = {"User-Agent": "Bob-agent (personal assistant)"}

_MODE_NAMES = {
    "BUS": "Bus", "TRAM": "Tram", "SUBWAY": "U-Bahn", "METRO": "S-Bahn",
    "REGIONAL_RAIL": "Regionalbahn", "REGIONAL_FAST_RAIL": "Regionalexpress",
    "HIGHSPEED_RAIL": "ICE", "LONG_DISTANCE": "Fernzug", "COACH": "Fernbus", "FERRY": "Fähre",
}


def _clock(iso: str) -> str:
    """'2026-10-03T18:30:00Z' → '20:30' in the user's timezone."""
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return moment.astimezone(ZoneInfo(settings.TIMEZONE)).strftime("%H:%M")


def _stop_name(place: dict, fallback: str) -> str:
    """The API calls the exact start/end coordinates 'START'/'END'."""
    return fallback if place["name"] in ("START", "END") else place["name"]


def _format_leg(leg: dict, origin: str, destination: str) -> str:
    start = _stop_name(leg["from"], origin)
    end = _stop_name(leg["to"], destination)
    minutes = round(leg["duration"] / 60)
    if leg["mode"] == "WALK":
        if start == end:
            return f"  - {minutes} Min. Umsteigen in {start}"
        return f"  - {minutes} Min. Fußweg: {start} → {end}"

    line = " ".join(filter(None, [_MODE_NAMES.get(leg["mode"], leg["mode"]), leg.get("routeShortName")]))
    text = f"  - {_clock(leg['startTime'])} {line}"
    if leg.get("headsign"):
        text += f" Richtung {leg['headsign']}"
    text += f": {start}"
    if leg["from"].get("track"):
        text += f" (Gleis {leg['from']['track']})"
    text += f" → {end}, an {_clock(leg['endTime'])}"
    delay = round((datetime.fromisoformat(leg["startTime"].replace("Z", "+00:00"))
                   - datetime.fromisoformat(leg["scheduledStartTime"].replace("Z", "+00:00"))).total_seconds() / 60)
    if delay > 0:
        text += f" (+{delay} Min. Verspätung)"
    return text


def _format_itinerary(number: int, itinerary: dict, origin: str, destination: str) -> str:
    """One connection as a few lines of text instead of ~30k characters of raw JSON."""
    transfers = itinerary["transfers"]
    header = (
        f"Verbindung {number}: ab {_clock(itinerary['startTime'])}, an {_clock(itinerary['endTime'])} "
        f"({round(itinerary['duration'] / 60)} Min., "
        f"{transfers} Umstieg{'' if transfers == 1 else 'e'})"
    )
    legs = [_format_leg(leg, origin, destination) for leg in itinerary["legs"]
            # Skip the one-minute shuffles inside a station; they only add noise.
            if not (leg["mode"] == "WALK" and leg["duration"] < 120)]
    return "\n".join([header, *legs])


def _find_place(name: str) -> dict | None:
    """Turn free text like 'Alexanderplatz, Berlin' into the best matching place."""
    response = httpx.get(f"{API_URL}/geocode",
                         params={"text": name}, headers=HEADERS, timeout=15)
    response.raise_for_status()
    results = response.json()
    # The top hit is often a long-distance coach stop with poor data. Prefer a stop that
    # local transit (bus, U-Bahn, S-Bahn, tram, train) actually serves.
    for place in results:
        if place["type"] == "STOP" and set(place.get("modes") or []) - {"COACH"}:
            return place
    return results[0] if results else None


@tool(parse_docstring=True)
def plan_journey(
    origin: str,
    destination: str,
    time: str | None = None,
    arrive_by: bool = False,
) -> str:
    """Find public transport connections (bus, train, U-Bahn, S-Bahn, tram) between two places.

    Use this when the user asks how to get somewhere, which train to take,
    or when they have to leave to arrive on time.

    Args:
        origin: Where the trip starts. Add the city, e.g. 'Alexanderplatz, Berlin'.
        destination: Where the trip ends. Add the city, e.g. 'Potsdamer Platz, Berlin'.
        time: Departure time in ISO format without timezone, e.g. '2026-09-26T14:00'. Leave empty for now.
        arrive_by: True if time is the latest arrival time instead of the departure time.
    """
    try:
        start_place = _find_place(origin)
        end_place = _find_place(destination)
    except httpx.HTTPError as e:
        return f"Error: the transport service is not reachable right now ({e})."
    if not start_place:
        return f"Error: could not find a place called '{origin}'. Ask the user to be more precise."
    if not end_place:
        return f"Error: could not find a place called '{destination}'. Ask the user to be more precise."

    params = {
        "fromPlace": f"{start_place['lat']},{start_place['lon']}",
        "toPlace": f"{end_place['lat']},{end_place['lon']}",
        "numItineraries": 3,
    }
    if time:
        try:
            params["time"] = _parse_local(time).isoformat()
        except ValueError:
            return f"Error: '{time}' is not a valid ISO date. Use e.g. '2026-09-26T14:00'."
        params["arriveBy"] = "true" if arrive_by else "false"

    try:
        response = httpx.get(
            f"{API_URL}/plan", params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"Error: the transport service is not reachable right now ({e})."

    itineraries = response.json().get("itineraries", [])
    if not itineraries:
        return f"No public transport connection found from {origin} to {destination}."

    # Results come sorted by departure. For "arrive by 9:00" the best ones are the LAST
    # (latest departure that still makes it); otherwise the first ones (leave soonest).
    chosen = itineraries[-3:] if arrive_by else itineraries[:3]

    return "\n\n".join(
        _format_itinerary(number, itinerary, origin, destination)
        for number, itinerary in enumerate(chosen, start=1)
    )


TRANSPORT_TOOLS = [plan_journey]
