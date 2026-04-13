from .graph import Graph
from .frontier import Frontier


def dfs_search(graph: Graph):
    start = graph.origin # Start node
    visited = set([start]) # Visited nodes
    frontier = Frontier('stack') # Container for dfs
    frontier.push((start, [start])) # Push start node to container
    nodes_created = 1 

    # Check if container is empty if not remove first node and return next
    while not frontier.is_empty():
        node_id, path = frontier.pop()
        print(f"Popped: {node_id}, frontier size before expansion: {frontier._container}") # Delete later


        # If node is destination return node
        if node_id in graph.destinations:
            return node_id, nodes_created, path

        # Retrieve neighbours' ids + cost in adjacency list -> neighbours = list of tuples
        neighbours = graph.adj.get(node_id, [])

        # Sort neighbours in ascending order
        sorted_neighbours = sorted(neighbours, key=lambda x: x[0])

        # Loop through reversed list cause LIFO
        for neighbour_id, edge_cost in reversed(sorted_neighbours):
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    new_path = path + [neighbour_id] 
                    nodes_created += 1
                    frontier.push((neighbour_id, new_path))
                    print(f"Pushed: {neighbour_id}, frontier size now: {frontier._container}") # Delete later

    return None, nodes_created, None

def bfs_search(graph: Graph):
    start = graph.origin # Start node
    visited = set([start]) # Visited nodes
    frontier = Frontier('queue') # Container for dfs
    frontier.push((start, [start])) # Push start node to container
    nodes_created = 1 

    # Check if container is empty if not remove first node and return next
    while not frontier.is_empty():
        node_id, path = frontier.pop()
        print(f"Popped: {node_id}, frontier size before expansion: {frontier._container}") # Delete later

    # If node is destination return node
        if node_id in graph.destinations:
            return node_id, nodes_created, path

        # Retrieve neighbours' ids + cost in adjacency list -> neighbours = list of tuples
        neighbours = graph.adj.get(node_id, [])

        # Sort neighbours in ascending order
        sorted_neighbours = sorted(neighbours, key=lambda x: x[0])

        # Loop through normal list cause FIFO
        for neighbour_id, edge_cost in sorted_neighbours:
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    new_path = path + [neighbour_id] 
                    nodes_created += 1
                    frontier.push((neighbour_id, new_path))
                    print(f"Pushed: {neighbour_id}, frontier size now: {frontier._container}") # Delete later


    return None, nodes_created, None