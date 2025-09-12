from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="nyc-housing-risk")

def address_to_location(address: str):
    """
    Convert a free-text address to (latitude, longitude, display_name).
    """
    try:
        location = geolocator.geocode(address)
        if location:
            return {
                "lat": location.latitude,
                "lon": location.longitude,
                "display_name": location.address
            }
        else:
            return None
    except Exception as e:
        print("Geocoding error:", e)
        return None
