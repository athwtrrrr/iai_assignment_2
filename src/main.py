import sys
from .graph import load_graph
from .search import dfs_search, bfs_search, a_star_search, gbfs_search, ucs_search, ids_search 

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

    if method == "DFS":
        goal, nodes_created, path = dfs_search(graph)
    elif method == "BFS":
        goal, nodes_created, path = bfs_search(graph)
    elif method == "A*":
        goal, nodes_created, path = a_star_search(graph)
    elif method == "GBFS":
        goal, nodes_created, path = gbfs_search(graph)
    elif method == "UCS":
        goal, nodes_created, path = ucs_search(graph)
    elif method == "IDS":
        goal, nodes_created, path = ids_search(graph)
    else:
        print(f"Error: Method '{method}' not supported yet.")
        sys.exit(1)

    if goal is None:
        print(f"{filename} {method} None {nodes_created} None")
    else:
        path_str = ' '.join(str(node) for node in path)
        print(f"{filename} {method} {goal} {nodes_created} {path_str}")

if __name__ == "__main__":
    main()