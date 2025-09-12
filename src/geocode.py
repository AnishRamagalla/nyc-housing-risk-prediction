# src/geocode.py
import requests

def geocode_address(address: str) -> tuple[float, float, str] | tuple[None, None, None]:
    """
    Forward geocode an NYC address using GeoSearch /v2/search.
    Returns (lat, lon, label) or (None, None, None) if not found.
    """
    url = "https://geosearch.planninglabs.nyc/v2/search"
    params = {"text": address, "size": 1}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get("features"):
            feat = data["features"][0]
            lon, lat = feat["geometry"]["coordinates"]
            label = feat["properties"].get("label", feat.get("place_name", address))
            return lat, lon, label
    except Exception as e:
        print(f"[WARN] GeoSearch failed for {address}: {e}")

    return None, None, None
