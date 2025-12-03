# src/utils/geo.py
from math import radians, sin, cos, asin, sqrt, degrees, atan2

class GeoUtils:
    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        """Tính khoảng cách giữa 2 điểm GPS theo đường chim bay (km)."""
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        return 2 * R * asin(sqrt(a))

    @staticmethod
    def dest_from(lat, lon, distance_km, bearing_deg):
        """Tính toạ độ đích từ điểm xuất phát, khoảng cách và góc phương vị."""
        R = 6371.0
        br = radians(bearing_deg)
        lat1, lon1 = radians(lat), radians(lon)
        lat2 = asin(sin(lat1) * cos(distance_km / R) + cos(lat1) * sin(distance_km / R) * cos(br))
        lon2 = lon1 + atan2(sin(br) * sin(distance_km / R) * cos(lat1),
                            cos(distance_km / R) - sin(lat1) * sin(lat2))
        return degrees(lat2), degrees(lon2)