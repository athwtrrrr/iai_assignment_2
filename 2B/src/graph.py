import pandas as pd
import networkx as nx
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance between two lat/lon points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def extract_road_name(location):
    if not location or not isinstance(location, str):
        return ""
    return location.split(" ")[0]


def build_graph(site_info_path="data/site_info.csv", max_distance_km=2.0):
    """
    Build the Boroondara road network as an undirected graph.

    Two SCATS sites are linked if their straight-line distance is at most
    max_distance_km (default 2 km). This yields a connected graph suitable
    for route search when official link tables are unavailable.
    """
    sites = pd.read_csv(site_info_path)
    sites["road"] = sites["location"].apply(extract_road_name)

    G = nx.Graph()
    for _, row in sites.iterrows():
        G.add_node(
            int(row["scats_id"]),
            lat=row["lat"],
            lon=row["lon"],
            location=row["location"],
            road=row["road"],
        )

    ids = sites["scats_id"].tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = sites.iloc[i]
            b = sites.iloc[j]
            dist = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if dist <= max_distance_km:
                G.add_edge(
                    int(a["scats_id"]),
                    int(b["scats_id"]),
                    distance_km=dist,
                    same_road=a["road"] == b["road"] and a["road"] != "",
                )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        print(f"  Warning: graph has {nx.number_connected_components(G)} components")
    return G, sites


if __name__ == "__main__":
    G, sites = build_graph()
