#!/usr/bin/env python3
"""
Traffic-Based Route Guidance System (TBRGS) — CLI entry point.
"""
import argparse
import os
import sys

import yaml

from route_search import format_route_summary, top_k_paths, find_best_path_a_star


def load_config(path="../config.yaml"):
    cfg_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    cfg = load_config()
    d = cfg.get("defaults", {})

    parser = argparse.ArgumentParser(description="TBRGS — Boroondara traffic route guidance")
    parser.add_argument("-o", "--origin", type=int, default=d.get("origin", 2000))
    parser.add_argument("-d", "--destination", type=int, default=d.get("destination", 3002))
    parser.add_argument("-t", "--timestamp", default=d.get("timestamp", "2006-10-27 08:00"))
    parser.add_argument("-m", "--model", choices=["lstm", "gru", "cnn"], default=d.get("model", "lstm"))
    parser.add_argument("-k", "--top-k", type=int, default=d.get("top_k", 5))
    parser.add_argument("--astar", action="store_true", help="Also run Part A A* on best path")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["route", "train-lstm", "train-gru", "train-cnn", "evaluate", "process-data"],
        default="route",
        help="Command (default: route)",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.command == "process-data":
        import data_processing

        data_processing  # noqa: B018 — run via __main__
        os.system(f"{sys.executable} data_processing.py")
        return

    if args.command == "train-lstm":
        os.system(f"{sys.executable} lstm.py")
        return
    if args.command == "train-gru":
        os.system(f"{sys.executable} gru.py")
        return
    if args.command == "train-cnn":
        os.system(f"{sys.executable} cnn.py")
        return
    if args.command == "evaluate":
        from evaluate import evaluate_all

        evaluate_all()
        return

    routes = top_k_paths(
        args.origin,
        args.destination,
        args.timestamp,
        k=args.top_k,
        model=args.model,
    )
    print(format_route_summary(routes, args.origin, args.destination))
    print(f"\nModel: {args.model.upper()}  |  Time: {args.timestamp}")

    if args.astar and routes:
        astar = find_best_path_a_star(args.origin, args.destination, args.timestamp, args.model)
        if astar:
            print(
                f"\nPart A A* confirms best route: {astar['total_sec']:.0f}s "
                f"({astar['nodes_expanded']} nodes expanded)"
            )


if __name__ == "__main__":
    main()
