import heapq
import math
import random

# Abstract class with Initial State, a way to move (Actions), and a way to guess distance (Heuristic)."
class SearchProblem:
    def __init__(self, initial, goal):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def heuristic(self, state):
        raise NotImplementedError


class DroneCityProblem(SearchProblem):
    def __init__(self, initial, goal, width, height, buildings): # Stores the map dimensions
        super().__init__(initial, goal)
        self.width = width
        self.height = height
        self.buildings = buildings  # Set of coordinates that are blocked

    def actions(self, state): #assuming 8 possible directions: E, W, N, S, NE, NW, SE, SW
        x, y = state
        moves = [
            ('N', (0, -1)), ('S', (0, 1)), ('E', (1, 0)), ('W', (-1, 0)),
            ('NE', (1, -1)), ('NW', (-1, -1)), ('SE', (1, 1)), ('SW', (-1, 1))
        ]

        valid_actions = []
        for action_name, (dx, dy) in moves:
            nx, ny = x + dx, y + dy
            # Check boundaries
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Check for collisions with buildings
                if (nx, ny) not in self.buildings:
                    valid_actions.append((action_name, (nx, ny)))
        return valid_actions

    def result(self, state, action_tuple):
        return action_tuple[1] # Action tuple is (name, new_coordinates)

    def heuristic(self, state): # Euclidean Distance for diagonal movement support (The Pythagorean theorem: a^2 + b^2 = c^2)
        x1, y1 = state
        x2, y2 = self.goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Greedy Best-First Search: Uses a Priority Queue sorted ONLY by h(n) (heuristic).
def greedy_best_first_search(problem):
    start_node = problem.initial
    frontier = []
    heapq.heappush(frontier, (problem.heuristic(start_node), start_node, []))

    explored = set()

    while frontier:
        # Pop the node that is estimated to be closest to the goal
        h, current_state, path = heapq.heappop(frontier)

        if current_state == problem.goal:
            return path, current_state

        if current_state in explored:
            continue
        explored.add(current_state)

        for action_name, next_state in problem.actions(current_state):
            if next_state not in explored:
                new_path = path + [next_state]  # Store coordinates in path for simpler printing
                h_new = problem.heuristic(next_state)
                heapq.heappush(frontier, (h_new, next_state, new_path))

    return None, None


def print_city_map(width, height, buildings, start, goal, path):
    print("\n--- City Drone Map ---")
    print("S: Start | G: Goal | [ ]: Building | * : Flight Path")

    path_set = set(path) if path else set()

    top_border = "xx" + "==" * width + "xx"
    print(top_border)

    for y in range(height):
        row = "||"
        for x in range(width):
            curr = (x, y)
            if curr == start:
                row += "S "
            elif curr == goal:
                row += "G "
            elif curr in buildings:
                row += "[]"
            elif curr in path_set:
                row += "* "
            else:
                row += ". "
        print(row + "||")

    print(top_border)


# --- Interactive Execution ---
if __name__ == "__main__":
    print("Greedy Best-First Search: Drone Flight Planning")

    try:
        # 1. Setup Grid
        w = 20
        h = 10

        # 2. User Input
        print(f"City Grid size is {w}x{h}")
        sx = int(input(f"Enter Drone Start X (0-{w - 1}): ") or 0)
        sy = int(input(f"Enter Drone Start Y (0-{h - 1}): ") or 0)
        gx = int(input(f"Enter Accident Site X (0-{w - 1}): ") or w - 1)
        gy = int(input(f"Enter Accident Site Y (0-{h - 1}): ") or h - 1)

        start = (sx, sy)
        goal = (gx, gy)

        # 3. Generate Random Skylines (Obstacles)
        # Creating "walls" of buildings to force the drone to navigate around
        buildings = set()
        for _ in range(30):
            bx = random.randint(0, w - 1)
            by = random.randint(0, h - 1)
            if (bx, by) != start and (bx, by) != goal:
                buildings.add((bx, by))

        # Add a specific wall to test navigation
        mid_x = w // 2
        for y in range(h // 4, h // 4 * 3):
            if (mid_x, y) != start and (mid_x, y) != goal:
                buildings.add((mid_x, y))

        problem = DroneCityProblem(start, goal, w, h, buildings)

        print("\nDrone is calculating path...")
        path, final = greedy_best_first_search(problem)

        if path:
            print(f"Path found! Length: {len(path)} steps.")
            print_city_map(w, h, buildings, start, goal, path)
        else:
            print("No path found. The goal is enclosed by buildings.")

    except ValueError:
        print("Please enter valid integers.")