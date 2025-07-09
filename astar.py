import heapq

class Node:
  def __init__(self, position, g, h):
    self.position = position  # (x, y) coordinates
    self.g = g  # Cost from start to node
    self.h = h  # Heuristic cost to goal
    self.f = g + h  # Total cost

  def __lt__(self, other):
    return self.f < other.f

def heuristic(a, b):
  return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(start, goal, grid):
  open_list = []
  closed_set = set()
  start_node = Node(start, 0, heuristic(start, goal))
  heapq.heappush(open_list, start_node)

  came_from = {}
  g_score = {start: 0}
  
  while open_list:
    current_node = heapq.heappop(open_list)

    if current_node.position == goal:
      path = []
      while current_node.position in came_from:
        path.append(current_node.position)
        current_node = came_from[current_node.position]
      return path[::-1]  # Return reversed path

    closed_set.add(current_node.position)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 possible directions
      neighbor = (current_node.position[0] + dx, current_node.position[1] + dy)

      if (0 <= neighbor[0] < len(grid) and
        0 <= neighbor[1] < len(grid[0]) and
        grid[neighbor[0]][neighbor[1]] == 0 and
        neighbor not in closed_set):

        tentative_g_score = g_score[current_node.position] + 1

        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
          came_from[neighbor] = current_node
          g_score[neighbor] = tentative_g_score
          h = heuristic(neighbor, goal)
          neighbor_node = Node(neighbor, tentative_g_score, h)

          if neighbor_node not in open_list:
            heapq.heappush(open_list, neighbor_node)

  return []  # Return empty if no path found