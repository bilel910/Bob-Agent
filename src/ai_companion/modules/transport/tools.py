from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from langchain_core.tools import tool

from ai_companion.modules.calendar import _parse_local
from ai_companion.settings import settings


API_URL = "https://api.transitous.org/api/v1"
# Transitous is a free community service: identify yourself so they can contact you.
HEADERS = {"User-Agent": "Bob-agent (personal assistant)"}


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

    return str(chosen)  # temporary: step 4 turns this into readable text


TRANSPORT_TOOLS = [plan_journey]
