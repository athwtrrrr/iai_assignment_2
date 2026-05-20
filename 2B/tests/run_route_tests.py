"""
Run 17 route scenarios and write results for the report (manual / integration tests).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))

from route_search import top_k_paths

TS = "2006-10-27 08:00"
CASES = [
    (2000, 3002),
    (2000, 3001),
    (3001, 3002),
    (3812, 3804),
    (970, 2000),
    (2200, 3120),
    (2820, 2825),
    (3122, 3126),
    (3180, 3127),
    (2000, 3120),
    (3002, 3126),
    (2825, 2827),
    (2846, 3001),
    (3120, 3122),
    (3812, 2000),
    (3001, 3812),
    (2200, 3002),
]

if __name__ == "__main__":
    out = []
    for o, d in CASES:
        routes = top_k_paths(o, d, TS, k=1, model="lstm")
        status = "OK" if routes else "NO_PATH"
        mins = routes[0]["total_sec"] / 60 if routes else None
        out.append({"origin": o, "dest": d, "status": status, "best_min": mins})
        print(f"{o} -> {d}: {status}" + (f" ({mins:.1f} min)" if mins else ""))

    import pandas as pd

    pd.DataFrame(out).to_csv("models/route_test_results.csv", index=False)
    print("\nWrote models/route_test_results.csv")
