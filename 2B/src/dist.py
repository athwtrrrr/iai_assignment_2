import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

sites = pd.read_csv("data/site_info.csv")
sites["road"] = sites["location"].apply(lambda loc: loc.split(" ")[0] if loc else "")

# For every pair of sites on the same road, compute distance
same_road_distances = []

for road, group in sites.groupby("road"):
    group = group.reset_index(drop=True)
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            dist = haversine(
                group.loc[i, "lat"], group.loc[i, "lon"],
                group.loc[j, "lat"], group.loc[j, "lon"]
            )
            same_road_distances.append({
                "road"    : road,
                "site_a"  : group.loc[i, "scats_id"],
                "site_b"  : group.loc[j, "scats_id"],
                "dist_km" : dist
            })

df_dist = pd.DataFrame(same_road_distances)
print(df_dist["dist_km"].describe())
print(f"\nUnder 0.5 km: {(df_dist['dist_km'] < 0.5).sum()}")
print(f"Under 1.0 km: {(df_dist['dist_km'] < 1.0).sum()}")
print(f"Under 1.5 km: {(df_dist['dist_km'] < 1.5).sum()}")
print(f"Under 2.0 km: {(df_dist['dist_km'] < 2.0).sum()}")
print(f"\nSmallest distances (likely adjacent sites):")
print(df_dist.sort_values("dist_km").head(20).to_string())