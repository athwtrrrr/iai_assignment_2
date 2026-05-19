import pandas as pd
import numpy as np
import networkx as nx
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance between two lat/lon points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


def extract_road_name(location):
    """
    Extract primary road name from location string.
    """
    return location.split(" ")[0] if location else ""


def build_graph(site_info_path="data/site_info.csv", max_distance_km=2.0):
    """
    Build road network as a networkx Graph.

    Two sites are connected if:
    1. They share the same primary road name
    2. They are within max_distance_km of each other

    Edge attributes:
        distance_km : float — haversine distance between sites
    """
    sites = pd.read_csv(site_info_path)
    sites["road"] = sites["location"].apply(extract_road_name)

    G = nx.Graph()

    # Add all sites as nodes
    for _, row in sites.iterrows():
        G.add_node(
            row["scats_id"],
            lat      = row["lat"],
            lon      = row["lon"],
            location = row["location"],
        )

    # Add edges between adjacent sites on the same road
    for i, site_a in sites.iterrows():
        for j, site_b in sites.iterrows():
            if j <= i:
                continue   # avoid duplicate pairs

            # Must share the same road name
            if site_a["road"] != site_b["road"]:
                continue

            # Must be close enough
            dist = haversine(
                site_a["lat"], site_a["lon"],
                site_b["lat"], site_b["lon"]
            )
            if dist <= max_distance_km:
                G.add_edge(
                    site_a["scats_id"],
                    site_b["scats_id"],
                    distance_km = dist
                )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, sites


if __name__ == "__main__":
    G, sites = build_graph()