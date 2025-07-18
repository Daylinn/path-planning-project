import heapq

class Node:
  def __init__(self, position, g, h):
    self.position = position  # (x, y) coordinates
    self.g = g  # Cost from start to node
    self.h = h  # Heuristic cost to goal
    self.f = g + h  # Total cost

  def __lt__(self, other):
    return self.f < other.f

def heuristic(a, b, method="manhattan"):
  if method == "manhattan":
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
  elif method == "euclidean":
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
  else:
    raise ValueError("Unsupported heuristic method.")

def astar(start, goal, grid):
  if start == goal:
    return [start]

  open_list = []
  closed_set = set()
  start_node = Node(start, 0, heuristic(start, goal, method="manhattan"))
  heapq.heappush(open_list, start_node)

  came_from = {}
  g_score = {start: 0}
  
  node_map = {start: start_node}

  while open_list:
    current_node = heapq.heappop(open_list)

    if current_node.position == goal:
      path = [goal]
      while path[-1] in came_from:
        path.append(came_from[path[-1]])
      return path[::-1]

    closed_set.add(current_node.position)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      neighbor = (current_node.position[0] + dx, current_node.position[1] + dy)

      if (0 <= neighbor[0] < len(grid) and
          0 <= neighbor[1] < len(grid[0]) and
          grid[neighbor[0]][neighbor[1]] == 0 and
          neighbor not in closed_set):

        tentative_g_score = g_score[current_node.position] + 1

        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          g_score[neighbor] = tentative_g_score
          came_from[neighbor] = current_node.position
          h = heuristic(neighbor, goal, method="manhattan")
          neighbor_node = Node(neighbor, tentative_g_score, h)
          heapq.heappush(open_list, neighbor_node)
          node_map[neighbor] = neighbor_node

  print("No path found from", start, "to", goal)
  return []