import math
from .graph import Graph
from .frontier import Frontier

def dfs_search(graph: Graph):
    start = graph.origin 
    visited = {start}
    frontier = Frontier('stack') 
    frontier.push((start, [start])) 
    nodes_created = 1 

    while not frontier.is_empty():
        node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        neighbours = graph.adj.get(node_id, [])
        for neighbour_id, _ in reversed(sorted(neighbours, key=lambda x: x[0])):
            if neighbour_id not in visited:
                visited.add(neighbour_id)
                nodes_created += 1
                frontier.push((neighbour_id, path + [neighbour_id]))
                
    return None, nodes_created, None

def bfs_search(graph: Graph):
    start = graph.origin 
    visited = {start}
    frontier = Frontier('queue') 
    frontier.push((start, [start])) 
    nodes_created = 1 

    while not frontier.is_empty():
        node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        neighbours = graph.adj.get(node_id, [])
        for neighbour_id, _ in sorted(neighbours, key=lambda x: x[0]):
            if neighbour_id not in visited:
                visited.add(neighbour_id)
                nodes_created += 1
                frontier.push((neighbour_id, path + [neighbour_id]))
                
    return None, nodes_created, None

# Đường chim bay
def heuristic(node_id: int, graph: Graph) -> float:
    if not graph.destinations:
        return 0.0
    node = graph.nodes[node_id]
    return min(math.sqrt((node.x - graph.nodes[d].x)**2 + (node.y - graph.nodes[d].y)**2) for d in graph.destinations)

# f = g + h
def a_star_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((heuristic(graph.origin, graph), graph.origin, [graph.origin], 0.0))
    explored = set()
    nodes_created = 1

    while not frontier.is_empty():
        _, node_id, path, g_cost = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        if node_id not in explored:
            explored.add(node_id)
            for n_id, cost in graph.adj.get(node_id, []):
                if n_id not in explored:
                    nodes_created += 1
                    new_g = g_cost + cost
                    frontier.push((new_g + heuristic(n_id, graph), n_id, path + [n_id], new_g))
                    
    return None, nodes_created, None

# Tương tự heuristic.
def gbfs_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((heuristic(graph.origin, graph), graph.origin, [graph.origin]))
    explored = set()
    nodes_created = 1

    while not frontier.is_empty():
        _, node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        if node_id not in explored:
            explored.add(node_id)
            for n_id, _ in graph.adj.get(node_id, []):
                if n_id not in explored:
                    nodes_created += 1
                    frontier.push((heuristic(n_id, graph), n_id, path + [n_id]))
                    
    return None, nodes_created, None

# Là A* nhưng h = 0, sử dụng g(cost) để mở rộng nút.
def ucs_search(graph: Graph):
    frontier = Frontier('priority')
    frontier.push((0.0, graph.origin, [graph.origin]))
    explored = set()
    nodes_created = 1

    while not frontier.is_empty():
        g_cost, node_id, path = frontier.pop()
        if node_id in graph.destinations:
            return node_id, nodes_created, path
            
        if node_id not in explored:
            explored.add(node_id)
            for n_id, cost in graph.adj.get(node_id, []):
                if n_id not in explored:
                    nodes_created += 1
                    frontier.push((g_cost + cost, n_id, path + [n_id]))
                    
    return None, nodes_created, None

# = BFS + DFS; chạy DFS lặp lại với giới hạn độ sâu tăng dần từ 0,1,2,... => save dung lượng như DFS nhưng vẫn đảm bảo tối ưu số bước như BFS.
def ids_search(graph: Graph): 
    def dls(node_id, limit, path, current_visited):
        nonlocal nodes_created
        if node_id in graph.destinations:
            return node_id, path
        if limit == 0:
            return None, None
            
        for n_id, _ in sorted(graph.adj.get(node_id, []), key=lambda x: x[0]):
            if n_id not in current_visited:
                current_visited.add(n_id)
                nodes_created += 1
                res_id, res_path = dls(n_id, limit - 1, path + [n_id], current_visited)
                if res_id is not None:
                    return res_id, res_path
                current_visited.remove(n_id)
        return None, None

    nodes_created = 1
    depth_limit = 0
    max_depth = len(graph.nodes) * 2 
    
    while depth_limit <= max_depth:
        goal_id, path = dls(graph.origin, depth_limit, [graph.origin], {graph.origin})
        if goal_id is not None:
            return goal_id, nodes_created, path
        depth_limit += 1

    return None, nodes_created, None