"""Geocode the institutions of the inventory: ``bids_inventory institutions update``.

The institution names of the inventory that are not in institutions.json yet
are geocoded with Nominatim (OpenStreetMap). A name located at the same address
as an institution already in the file is added to its alternative names,
otherwise it is added as a new institution.
Names that could not be located are added to the "unlocated" entry.

When a name has a country code (DICOM addresses end with it: "...,Magdeburg,District,DE,39120"),
it is only searched in that country: otherwise parts of the address,
for example "IBMI", can be found anywhere in the world.
If it has a ZIP code too, the address found must be near it,
otherwise the institution is located at its ZIP code.
Results that are only a country or a region are not used.

``bids_inventory institutions relocate`` geocodes again institutions found in the wrong place.

Needs the 'institutions' extra.
"""

from typing import Any

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from geopy.location import Location

from bids_inventory._institutions_matching import distance_km
from bids_inventory.institutions import (
    UNLOCATED,
    clean_address,
    country_code,
    known_names,
    new_institution,
    postal_code,
    save_institutions,
)
from bids_inventory.logger import bi_logger

bi_log = bi_logger()


def update_institutions(
    institutions: dict[str, dict[str, Any]], names: Any, debug: bool = False
) -> dict[str, dict[str, Any]]:
    """Add the institution names that are not in the file yet.

    The institutions already in the file are not changed,
    except to add alternative names.
    The file is saved after each geocoded address:
    if the update is stopped, running it again continues where it stopped.

    Parameters
    ----------
    debug : bool
        Only geocode a few addresses.
    """
    known = known_names(institutions)
    new_names = sorted({x for x in names if x not in known})

    # several new names can have the same address once cleaned: geocode it once
    by_address: dict[str, list[str]] = {}
    for name in new_names:
        by_address.setdefault(clean_address(name), []).append(name)
    to_geocode = sorted(by_address)
    bi_log.info(f"{len(known)} institution names known, {len(to_geocode)} new addresses to geocode")
    if debug:
        to_geocode = to_geocode[:3]

    geocode = _geocoder()
    for address in to_geocode:
        bi_log.info(address)
        found = _locate(
            geocode,
            address,
            _names_country(by_address[address]),
            postal_code(by_address[address][0]),
        )
        add_names(institutions, by_address[address], found)
        save_institutions(institutions)

    save_institutions(institutions)
    return institutions


def relocate_institutions(
    institutions: dict[str, dict[str, Any]],
    names: list[str] | None = None,
    country: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Geocode again institutions, only in their country.

    Parameters
    ----------
    names : list[str] | None
        Institutions to geocode again (keys of institutions.json).
        If None: all the institutions with a country code in one of their names,
        and only those found in that country but located in another one,
        or far from their ZIP code, are changed.
    country : str | None
        Country code ("ch") to search the institutions in,
        for names without one (for example "University of Geneva").

    An institution not found in its country is moved to the "unlocated" entry,
    one found at the address of another institution is merged with it.
    """
    check_all = names is None
    if names is None:
        names = sorted(
            name
            for name, institution in institutions.items()
            if name != UNLOCATED and _names_country([name, *institution["AlternativeNames"]])
        )
    if unknown := sorted(set(names) - set(institutions)):
        raise ValueError(f"Not in institutions.json: {unknown}")

    geocode = _geocoder()
    for name in names:
        institution = institutions[name]
        all_names = [name, *institution["AlternativeNames"]]
        code = country or _names_country(all_names)
        if code is None:
            bi_log.warning(f"No country code for {name!r}: use --country")
            continue
        bi_log.info(f"{name} ({code})")
        found = _locate(geocode, clean_address(name), code, postal_code(name))
        if check_all and (not found or _same_place(found, institution)):
            # already in the right place, or the country code may not be one:
            # for example "Birmingham, AL, 35233" (Alabama, or Albania?)
            continue
        bi_log.info(f"  was: {institution['Address']}")
        del institutions[name]
        add_names(institutions, all_names, found)
        save_institutions(institutions)

    save_institutions(institutions)
    return institutions


# an institution further than this from the place found is in the wrong place
MAX_DISTANCE_KM = 50


def _same_place(found: dict[str, Any], institution: dict[str, Any]) -> bool:
    """Return whether the location found is in the same country and near the one in the file."""
    if found.get("country") != institution["Country"]:
        return False
    distance = distance_km(
        {"Latitude": found["latitude"], "Longitude": found["longitude"]}, institution
    )
    return distance is None or distance < MAX_DISTANCE_KM


def _names_country(names: list[str]) -> str | None:
    """Return the country code found in some names of the same institution, if any."""
    codes = {code for name in names if (code := country_code(name))}
    # several codes: probably not country codes after all
    return codes.pop() if len(codes) == 1 else None


def _geocoder() -> RateLimiter:
    geolocator = Nominatim(user_agent="bids-inventory-institutions", timeout=10)
    # Nominatim usage policy: at most 1 request per second
    return RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2)


def _locate(
    geocode: RateLimiter, address: str, country: str | None, postcode: str | None = None
) -> dict[str, Any]:
    """Geocode an address: empty if not found.

    Only in this country if one is given, and near its ZIP code if one is given:
    otherwise at the ZIP code.
    """
    location = _get_location(geocode, address, country)
    if postcode and country and (at_postcode := _postcode_location(geocode, postcode, country)):
        if location is None or _distance(location, at_postcode) > MAX_DISTANCE_KM:
            if location is not None:
                bi_log.info(f"  far from its ZIP code: {location.address}")
            location = at_postcode
    if location is None:
        bi_log.info("  not found")
        return {}
    bi_log.info(f"  {location.address}")
    details = location.raw.get("address", {})
    return {
        "address": location.address,
        "city": details.get("city") or details.get("town") or details.get("village"),
        "country": details.get("country"),
        "latitude": location.latitude,
        "longitude": location.longitude,
    }


def add_names(
    institutions: dict[str, dict[str, Any]], names: list[str], location: dict[str, Any]
) -> None:
    """Add names found at a location.

    To the institution already at this address if there is one,
    otherwise as a new institution.
    Names that could not be located are added to the "unlocated" entry.
    """
    if not location.get("address"):
        unlocated = institutions.setdefault(UNLOCATED, new_institution())
        unlocated["AlternativeNames"] = sorted(set(unlocated["AlternativeNames"]) | set(names))
        return
    same_address = [
        name
        for name, institution in institutions.items()
        if institution["Address"] == location["address"]
    ]
    if same_address:
        name = same_address[0]
    else:
        name, *names = names
        institutions[name] = new_institution(location)
    alternatives = set(institutions[name]["AlternativeNames"]) | set(names)
    institutions[name]["AlternativeNames"] = sorted(alternatives - {name})


def _distance(a: Location, b: Location) -> float:
    return (
        distance_km(
            {"Latitude": a.latitude, "Longitude": a.longitude},
            {"Latitude": b.latitude, "Longitude": b.longitude},
        )
        or 0.0
    )


def _postcode_location(geocode: RateLimiter, postcode: str, country: str) -> None | Location:
    """Geocode a ZIP code, or its beginning ("K7L" for "K7L3N6"): not all are known."""
    for candidate in dict.fromkeys([postcode, postcode.split()[0], postcode[:3]]):
        try:
            location = geocode(
                {"postalcode": candidate}, addressdetails=True, country_codes=country
            )
        except Exception as e:
            bi_log.warning(f"Could not geocode the ZIP code {candidate!r}: {e!r}")
            continue
        if location and not _is_vague(location):
            return location
    return None


def _is_vague(location: Location) -> bool:
    """Return whether the location is only a country or a region (useless for an institution)."""
    # Nominatim ranks: 4 for countries, 8 for states
    return int(location.raw.get("place_rank", 30)) <= 8


def _get_location(
    geocode: RateLimiter, address: str, country: str | None = None
) -> None | Location:
    """Geocode an address, then parts of it if it is not found.

    Only in this country (code, for example "de") if one is given.
    """
    if not address:
        return None
    options: dict[str, Any] = {"addressdetails": True}
    if country:
        options["country_codes"] = country
    try:
        if (location := geocode(address, **options)) and not _is_vague(location):
            return location
    except Exception as e:
        bi_log.warning(f"Could not geocode {address!r}: {e!r}")
    tokens = [x.strip() for x in address.split(",")]
    if len(tokens) > 1:
        # try without the first part (often a department), then only with it
        return _get_location(geocode, ", ".join(tokens[1:]), country) or _get_location(
            geocode, tokens[0], country
        )
    return None
