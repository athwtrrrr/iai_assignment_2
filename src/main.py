import sys
from .graph import load_graph
from .search import dfs_search, bfs_search, a_star_search, gbfs_search, ucs_search, ida_star_search
from .multi_goals_search import multi_goal_search

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.main <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    try:
        graph = load_graph(filename)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)

    # Choose the appropriate search
    if method == "DFS":
        goal, nodes_created, path = dfs_search(graph)
    elif method == "BFS":
        goal, nodes_created, path = bfs_search(graph)
    elif method == "AS":
        goal, nodes_created, path = a_star_search(graph)
    elif method == "GBFS":
        goal, nodes_created, path = gbfs_search(graph)
    elif method == "UCS":
        goal, nodes_created, path = ucs_search(graph)
    elif method == "IDA":
        goal, nodes_created, path = ida_star_search(graph)
    elif method == "MULTI":
        goal, nodes_created, path = multi_goal_search(graph)
    else:
        print(f"Error: Method '{method}' not supported.")
        sys.exit(1)

    if goal is None:
        print(f"{filename} {method}")
        print("No path found.")
    else:
        if method == "MULTI":
            print(f"{filename} {method}")
            print(f"{goal} {nodes_created}")
            # Mark visited destinations with *
            path_str = " -> ".join(
                f"{node}*" if node in graph.destinations else str(node)
                for node in path
            )
            print(path_str)
        else:
            path_str = " -> ".join(str(node) for node in path)
            print(f"{filename} {method}")
            print(f"{goal} {nodes_created}")
            print(path_str)

if __name__ == "__main__":
    main()