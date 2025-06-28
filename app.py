from flask import Flask, render_template, request, jsonify
import heapq
import random
import math

app = Flask(__name__)

# Sample cities and coordinates (for A* heuristic)
cities = {
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Gujarat": (22.3094, 72.1362),
    "Goa": (15.2993, 74.1240),
    "Kanpur": (26.4499, 80.3319),
    "Jammu": (32.7266, 74.8570),
    "Hyderabad": (17.3850, 78.4867),
    "Bangalore": (12.9716, 77.5946),
    "Gangtok": (27.3314, 88.6138),
    "Meghalaya": (25.4670, 91.3662)
}

def generate_graph(V=6):
    nodes = random.sample(list(cities.keys()), V)
    graph = {node: [] for node in nodes}

    # Connect nodes linearly first
    for i in range(1, V):
        u, v = nodes[i - 1], nodes[i]
        w = random.randint(30, 80)
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Add some extra edges
    for _ in range(V // 2):
        u, v = random.sample(nodes, 2)
        if u != v:
            w = random.randint(10, 50)
            graph[u].append((v, w))
            graph[v].append((u, w))

    return nodes, graph

def dijkstra(graph, src):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[src] = 0
    visited = set()
    pq = [(0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, prev

def heuristic(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star(graph, start, goal):
    open_set = [(0 + heuristic(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current == goal:
            break

        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return g_score, came_from

def reconstruct_path(prev, dst):
    path = []
    while dst:
        path.append(dst)
        dst = prev[dst]
    return path[::-1]

def prim_mst(graph):
    start = list(graph.keys())[0]
    visited = set([start])
    edges = []
    min_heap = [(w, start, v) for v, w in graph[start]]
    heapq.heapify(min_heap)
    mst_edges = []

    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            for to, wt in graph[v]:
                if to not in visited:
                    heapq.heappush(min_heap, (wt, v, to))
    return mst_edges

@app.route("/", methods=["GET", "POST"])
def index():
    nodes, graph = generate_graph()
    src = nodes[0]
    dst = nodes[-1]
    algo = "dijkstra"
    path = []
    total_time = None

    if request.method == "POST":
        src = request.form["src"]
        dst = request.form["dst"]
        algo = request.form["algo"]

    if algo == "dijkstra":
        dist, prev = dijkstra(graph, src)
        path = reconstruct_path(prev, dst)
        total_time = dist[dst]
    elif algo == "astar":
        dist, prev = a_star(graph, src, dst)
        path = reconstruct_path(prev, dst)
        total_time = dist[dst]

    mst = prim_mst(graph)

    return render_template("index.html", nodes=nodes, graph=graph, path=path, total_time=total_time,
                           src=src, dst=dst, algo=algo, mst=mst)

@app.route("/api/graph")
def api_graph():
    nodes, graph = generate_graph()
    return jsonify(graph)

if __name__ == "__main__":
    app.run(debug=True , port = 5000)
