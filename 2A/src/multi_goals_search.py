import heapq
from .graph import Graph


def multi_goal_search(graph: Graph):
    
    destinations = frozenset(graph.destinations) # Immutable destination set
    start_visited = frozenset({graph.origin} & destinations) # Check if origin is also a destination

    # (g_cost, tiebreak, node_id, visited_destinations, path)
    counter = 0
    frontier = [(0.0, counter, graph.origin, start_visited, [graph.origin])] 
    best_cost = {(graph.origin, start_visited): 0.0} # cheapest cost to reach a state
    nodes_created = 1

    while frontier:
        g_cost, _, node_id, visited_dests, path = heapq.heappop(frontier) # remove and return lowest g cost

        # goal check (only stop when all destinations are visited)
        if visited_dests == destinations:
            return node_id, nodes_created, path

        # cost check to skip expensive duplicate
        if g_cost > best_cost.get((node_id, visited_dests), float('inf')):
            continue
        
        # expand neighbours
        for n_id, edge_cost in sorted(graph.adj.get(node_id, []), key=lambda x: x[0]):
            new_g = g_cost + edge_cost # cost to reach neighbour via current path
            new_visited = visited_dests | (frozenset({n_id}) & destinations) # update visited list
            new_state = (n_id, new_visited) 

            if new_g < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_g
                nodes_created += 1
                counter += 1
                heapq.heappush(frontier, (new_g, counter, n_id, new_visited, path + [n_id]))

    return None, nodes_created, None