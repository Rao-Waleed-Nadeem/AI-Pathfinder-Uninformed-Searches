"""
AI Pathfinder Uninformed Search Visualization
Author: Rao Waleed Nadeem (23F-0628) 
This application visualizes various uninformed search algorithms (BFS, DFS, UCS, DLS, IDDFS, Bidirectional) on a grid-based environment. Users can draw walls, set start and target points, and watch how each algorithm explores the grid to find a path from start to target.
Key Features:
- Interactive grid where users can place walls, start (S), and target (T) points
- Multiple algorithms to choose from, each with its own exploration strategy
- Real-time visualization of explored nodes, frontier, and final path
Instructions:
1. Click on the grid to draw walls (dark cells) that block movement.
2. Use the "Start" and "Target" buttons to place the S and T points
3. Select an algorithm from the dropdown menu.
4. Press "RUN" to see the algorithm in action. Watch how it explores the grid and finds a path.
5. Press "RESET" to clear the grid and start over.
"""

import tkinter as tk
from tkinter import ttk
import heapq
import time
from collections import deque

# ─────────────────────────────────────────────
#  GRID SETTINGS
# ─────────────────────────────────────────────
ROWS = 15          # Number of rows in the grid
COLS = 20          # Number of columns in the grid
CELL  = 40         # Pixel size of each cell
DELAY = 80         # Milliseconds between each animation step

# ─────────────────────────────────────────────
#  COLORS  
# ─────────────────────────────────────────────
COLOR = {
    "empty":    "#FFFFFF",   # white      – free cell
    "wall":     "#2C3E50",   # dark       – static wall
    "start":    "#2ECC71",   # green      – start point S
    "target":   "#E74C3C",   # red        – target point T
    "frontier": "#F39C12",   # orange     – in queue/stack (waiting)
    "explored": "#AED6F1",   # light blue – already visited
    "path":     "#8E44AD",   # purple     – final path
}

# ─────────────────────────────────────────────
#  MOVEMENT DIRECTIONS  (strictly as required)
#  Clockwise order: Up, Right, Bottom,
#  Bottom-Right, Left, Top-Left
#  NOTE: Top-Right and Bottom-Left are NOT included
# ─────────────────────────────────────────────
DIRECTIONS = [
    (-1,  0),   # 1. Up
    ( 0,  1),   # 2. Right
    ( 1,  0),   # 3. Bottom
    ( 1,  1),   # 4. Bottom-Right (diagonal allowed)
    ( 0, -1),   # 5. Left
    (-1, -1),   # 6. Top-Left    (diagonal allowed)
]

# ─────────────────────────────────────────────────────────────────
#  HELPER: check if (r, c) is inside the grid
# ─────────────────────────────────────────────────────────────────
def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

# ─────────────────────────────────────────────────────────────────
#  HELPER: reconstruct path from 'came_from' dictionary
#  came_from stores: child -> parent  for each visited node
# ─────────────────────────────────────────────────────────────────
def reconstruct_path(came_from, start, goal):
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()               # path goes from start -> goal
    return path if path[0] == start else []

# ─────────────────────────────────────────────────────────────────
#  HELPER: get valid neighbors of a cell (no walls, inside grid)
# ─────────────────────────────────────────────────────────────────
def get_neighbors(r, c, grid):
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc) and grid[nr][nc] != "wall":
            neighbors.append((nr, nc))
    return neighbors

# =================================================================
#  SEARCH ALGORITHMS  (each is a generator that yields steps)
#  Yields:
#    ("step", explored_set, frontier_set)  <- during search
#    ("path", path_list)                   <- final result
# =================================================================

def bfs(grid, start, goal):
    """Breadth-First Search – explores level by level (shortest path)"""
    print("\n[BFS] Starting Breadth-First Search...")
    queue = deque([start])          # FIFO queue
    came_from = {start: None}
    explored = set()

    while queue:
        node = queue.popleft()      # take from front (FIFO)
        if node == goal:
            path = reconstruct_path(came_from, start, goal)
            print(f"[BFS] Path found! Length = {len(path)}")
            yield "path", path
            return
        if node in explored:
            continue
        explored.add(node)
        r, c = node
        for nr, nc in get_neighbors(r, c, grid):
            if (nr, nc) not in came_from:
                came_from[(nr, nc)] = node
                queue.append((nr, nc))

        frontier = set(queue)
        yield "step", explored.copy(), frontier.copy()

    print("[BFS] No path found!")
    yield "path", []


def dfs(grid, start, goal):
    """Depth-First Search – goes as deep as possible before backtracking"""
    print("\n[DFS] Starting Depth-First Search...")
    stack = [start]                 # LIFO stack
    came_from = {start: None}
    explored = set()

    while stack:
        node = stack.pop()          # take from top (LIFO)
        if node == goal:
            path = reconstruct_path(came_from, start, goal)
            print(f"[DFS] Path found! Length = {len(path)}")
            yield "path", path
            return
        if node in explored:
            continue
        explored.add(node)
        r, c = node
        for nr, nc in get_neighbors(r, c, grid):
            if (nr, nc) not in came_from:
                came_from[(nr, nc)] = node
                stack.append((nr, nc))

        frontier = set(stack)
        yield "step", explored.copy(), frontier.copy()

    print("[DFS] No path found!")
    yield "path", []


def ucs(grid, start, goal):
    """Uniform-Cost Search – always expands the cheapest (lowest cost) node"""
    print("\n[UCS] Starting Uniform-Cost Search...")
    # heap stores: (cost, node)
    heap = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    explored = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node == goal:
            path = reconstruct_path(came_from, start, goal)
            print(f"[UCS] Path found! Cost = {cost}, Length = {len(path)}")
            yield "path", path
            return
        if node in explored:
            continue
        explored.add(node)
        r, c = node
        for nr, nc in get_neighbors(r, c, grid):
            # diagonal move costs 1.4, straight costs 1.0
            dr = abs(nr - r)
            dc = abs(nc - c)
            step_cost = 1.4 if (dr + dc == 2) else 1.0
            new_cost = cost + step_cost
            if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                cost_so_far[(nr, nc)] = new_cost
                came_from[(nr, nc)] = node
                heapq.heappush(heap, (new_cost, (nr, nc)))

        frontier = {item[1] for item in heap}
        yield "step", explored.copy(), frontier.copy()

    print("[UCS] No path found!")
    yield "path", []


def dls(grid, start, goal, depth_limit=12):
    """Depth-Limited Search – DFS but stops at a maximum depth"""
    print(f"\n[DLS] Starting Depth-Limited Search (limit={depth_limit})...")
    # stack stores: (node, depth, path_so_far)
    stack = [(start, 0, [start])]
    explored = set()
    came_from = {start: None}

    while stack:
        node, depth, path_so_far = stack.pop()
        if node == goal:
            print(f"[DLS] Path found! Length = {len(path_so_far)}")
            yield "path", path_so_far
            return
        if depth >= depth_limit:
            continue            # hit the limit, don't go deeper
        if node in explored:
            continue
        explored.add(node)
        r, c = node
        for nr, nc in get_neighbors(r, c, grid):
            if (nr, nc) not in explored:
                came_from[(nr, nc)] = node
                stack.append(((nr, nc), depth + 1, path_so_far + [(nr, nc)]))

        frontier = {item[0] for item in stack}
        yield "step", explored.copy(), frontier.copy()

    print("[DLS] No path found within depth limit!")
    yield "path", []


def iddfs(grid, start, goal):
    """Iterative Deepening DFS – runs DLS with increasing depth limits"""
    print("\n[IDDFS] Starting Iterative Deepening DFS...")
    max_possible = ROWS * COLS     # absolute maximum depth

    for limit in range(1, max_possible + 1):
        print(f"  [IDDFS] Trying depth limit = {limit}")
        stack = [(start, 0, [start])]
        explored = set()

        while stack:
            node, depth, path_so_far = stack.pop()
            if node == goal:
                print(f"[IDDFS] Path found at depth {limit}! Length = {len(path_so_far)}")
                yield "path", path_so_far
                return
            if depth >= limit:
                continue
            if node in explored:
                continue
            explored.add(node)
            r, c = node
            for nr, nc in get_neighbors(r, c, grid):
                if (nr, nc) not in explored:
                    stack.append(((nr, nc), depth + 1, path_so_far + [(nr, nc)]))

            frontier = {item[0] for item in stack}
            yield "step", explored.copy(), frontier.copy()

    print("[IDDFS] No path found!")
    yield "path", []


def bidirectional(grid, start, goal):
    """Bidirectional Search – searches from both start AND goal simultaneously"""
    print("\n[Bidirectional] Starting Bidirectional Search...")

    # Forward search from start
    fwd_queue   = deque([start])
    fwd_visited = {start: None}      # node -> parent
    fwd_explored = set()

    # Backward search from goal
    bwd_queue   = deque([goal])
    bwd_visited = {goal: None}       # node -> parent
    bwd_explored = set()

    meeting_point = None

    while fwd_queue or bwd_queue:
        # ── Forward step ──
        if fwd_queue:
            node = fwd_queue.popleft()
            fwd_explored.add(node)
            # check if forward has reached backward's visited set
            if node in bwd_visited:
                meeting_point = node
                break
            r, c = node
            for nr, nc in get_neighbors(r, c, grid):
                if (nr, nc) not in fwd_visited:
                    fwd_visited[(nr, nc)] = node
                    fwd_queue.append((nr, nc))

        # ── Backward step ──
        if bwd_queue:
            node = bwd_queue.popleft()
            bwd_explored.add(node)
            # check if backward has reached forward's visited set
            if node in fwd_visited:
                meeting_point = node
                break
            r, c = node
            for nr, nc in get_neighbors(r, c, grid):
                if (nr, nc) not in bwd_visited:
                    bwd_visited[(nr, nc)] = node
                    bwd_queue.append((nr, nc))

        combined_explored = fwd_explored | bwd_explored
        combined_frontier = set(fwd_queue) | set(bwd_queue)
        yield "step", combined_explored, combined_frontier

    if meeting_point:
        # Build path: start -> meeting_point (forward)
        path_fwd = []
        node = meeting_point
        while node is not None:
            path_fwd.append(node)
            node = fwd_visited.get(node)
        path_fwd.reverse()

        # Build path: meeting_point -> goal (backward, reversed)
        path_bwd = []
        node = bwd_visited.get(meeting_point)
        while node is not None:
            path_bwd.append(node)
            node = bwd_visited.get(node)

        full_path = path_fwd + path_bwd
        print(f"[Bidirectional] Meeting at {meeting_point}. Path length = {len(full_path)}")
        yield "path", full_path
    else:
        print("[Bidirectional] No path found!")
        yield "path", []


# =================================================================
#  MAIN GUI APPLICATION
# =================================================================
class AIPathfinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Pathfinder – Uninformed Search Visualization")
        self.root.configure(bg="#1A1A2E")

        # Grid state: each cell is "empty", "wall", "start", "target"
        self.grid = [["empty"] * COLS for _ in range(ROWS)]

        # Start and Target positions
        self.start  = (0, 0)
        self.target = (ROWS - 1, COLS - 1)
        self.grid[self.start[0]][self.start[1]]   = "start"
        self.grid[self.target[0]][self.target[1]] = "target"

        # Track which mode user is drawing in
        self.draw_mode = "wall"       # "wall", "start", "target", "erase"
        self.running   = False        # is animation running?
        self.generator = None         # current search generator
        self.current_algo = "BFS"

        # Store rectangle IDs for each cell
        self.rects = {}
        self.step_count = 0

        self._build_ui()
        self._draw_grid()

        print("=" * 50)
        print("  AI Pathfinder – Q7 Solution")
        print("=" * 50)
        print("Instructions:")
        print("  • Click cells to draw WALLS")
        print("  • Select S/T buttons to place Start/Target")
        print("  • Choose an algorithm and press RUN")
        print("=" * 50)

    # ──────────────────────────────────────────
    #  BUILD THE UI (title, controls, canvas)
    # ──────────────────────────────────────────
    def _build_ui(self):
        # ── Title Bar ──
        title = tk.Label(
            self.root, text="AI Pathfinder – Uninformed Search",
            font=("Helvetica", 18, "bold"),
            bg="#1A1A2E", fg="#F0F0F0", pady=8
        )
        title.pack(fill=tk.X)

        subtitle = tk.Label(
            self.root, text="AI Pathfinder – Uninformed Search Visualization",
            font=("Helvetica", 10), bg="#1A1A2E", fg="#AED6F1"
        )
        subtitle.pack()

        # ── Control Panel ──
        ctrl = tk.Frame(self.root, bg="#16213E", pady=6)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        # Algorithm selector
        tk.Label(ctrl, text="Algorithm:", bg="#16213E", fg="white",
                 font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5)

        self.algo_var = tk.StringVar(value="BFS")
        algos = ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"]
        algo_menu = ttk.Combobox(ctrl, textvariable=self.algo_var,
                                 values=algos, width=14, state="readonly")
        algo_menu.grid(row=0, column=1, padx=5)

        # Drawing mode buttons
        tk.Label(ctrl, text="Draw:", bg="#16213E", fg="white",
                 font=("Helvetica", 10, "bold")).grid(row=0, column=2, padx=5)

        tk.Button(ctrl, text="Wall",   bg="#2C3E50", fg="white",
                  command=lambda: self._set_mode("wall")).grid(row=0, column=3, padx=3)
        tk.Button(ctrl, text="Start",  bg="#27AE60", fg="white",
                  command=lambda: self._set_mode("start")).grid(row=0, column=4, padx=3)
        tk.Button(ctrl, text="Target", bg="#C0392B", fg="white",
                  command=lambda: self._set_mode("target")).grid(row=0, column=5, padx=3)
        tk.Button(ctrl, text="Erase",  bg="#7F8C8D", fg="white",
                  command=lambda: self._set_mode("erase")).grid(row=0, column=6, padx=3)

        # Run / Reset buttons
        tk.Button(ctrl, text="RUN",   bg="#8E44AD", fg="white",
                  font=("Helvetica", 10, "bold"),
                  command=self._start_search).grid(row=0, column=7, padx=8)
        tk.Button(ctrl, text="RESET", bg="#E67E22", fg="white",
                  font=("Helvetica", 10, "bold"),
                  command=self._reset).grid(row=0, column=8, padx=3)

        # Mode label
        self.mode_label = tk.Label(
            ctrl, text="Mode: WALL", font=("Helvetica", 9, "bold"),
            bg="#16213E", fg="#F39C12"
        )
        self.mode_label.grid(row=0, column=9, padx=10)

        # ── Status Bar ──
        self.status_var = tk.StringVar(value="Click cells to draw walls. Press RUN to start!")
        status = tk.Label(self.root, textvariable=self.status_var,
                          font=("Helvetica", 9), bg="#0F3460", fg="#E0E0E0",
                          anchor="w", padx=10)
        status.pack(fill=tk.X, padx=10)

        # ── Canvas (Grid) ──
        canvas_frame = tk.Frame(self.root, bg="#1A1A2E")
        canvas_frame.pack(padx=10, pady=5)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=COLS * CELL + 1,
            height=ROWS * CELL + 1,
            bg="#FFFFFF", cursor="crosshair"
        )
        self.canvas.pack()

        # Mouse events for drawing walls / placing S and T
        self.canvas.bind("<Button-1>",  self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # ── Legend ──
        legend = tk.Frame(self.root, bg="#1A1A2E")
        legend.pack(pady=4)
        items = [
            ("Start (S)",  COLOR["start"]),
            ("Target (T)", COLOR["target"]),
            ("Wall",       COLOR["wall"]),
            ("Frontier",   COLOR["frontier"]),
            ("Explored",   COLOR["explored"]),
            ("Final Path", COLOR["path"]),
        ]
        for i, (label, color) in enumerate(items):
            box = tk.Label(legend, bg=color, width=2, relief="raised")
            box.grid(row=0, column=i * 2, padx=2)
            tk.Label(legend, text=label, bg="#1A1A2E", fg="white",
                     font=("Helvetica", 8)).grid(row=0, column=i * 2 + 1, padx=4)

        # ── Stats Bar ──
        self.stats_var = tk.StringVar(value="Steps: 0 | Explored: 0 | Path: -")
        stats = tk.Label(self.root, textvariable=self.stats_var,
                         font=("Helvetica", 9, "bold"), bg="#1A1A2E", fg="#F39C12")
        stats.pack(pady=2)

    # ──────────────────────────────────────────
    #  DRAW THE GRID ON CANVAS
    # ──────────────────────────────────────────
    def _draw_grid(self):
        self.canvas.delete("all")
        self.rects = {}
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = c * CELL, r * CELL
                x2, y2 = x1 + CELL, y1 + CELL
                cell_type = self.grid[r][c]
                fill = COLOR.get(cell_type, COLOR["empty"])
                rid = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=fill, outline="#BDC3C7", width=1
                )
                self.rects[(r, c)] = rid
                # Label S and T on the grid
                if cell_type == "start":
                    self.canvas.create_text(
                        x1 + CELL // 2, y1 + CELL // 2, text="S",
                        font=("Helvetica", 12, "bold"), fill="white"
                    )
                elif cell_type == "target":
                    self.canvas.create_text(
                        x1 + CELL // 2, y1 + CELL // 2, text="T",
                        font=("Helvetica", 12, "bold"), fill="white"
                    )

    # ──────────────────────────────────────────
    #  UPDATE A SINGLE CELL COLOR
    # ──────────────────────────────────────────
    def _color_cell(self, r, c, color_key):
        if (r, c) in self.rects:
            self.canvas.itemconfig(self.rects[(r, c)], fill=color_key)

    # ──────────────────────────────────────────
    #  MOUSE CLICK / DRAG to draw on grid
    # ──────────────────────────────────────────
    def _cell_from_event(self, event):
        c = event.x // CELL
        r = event.y // CELL
        if in_bounds(r, c):
            return r, c
        return None

    def _on_click(self, event):
        if self.running:
            return
        pos = self._cell_from_event(event)
        if pos:
            self._apply_draw(pos)

    def _on_drag(self, event):
        if self.running:
            return
        pos = self._cell_from_event(event)
        if pos:
            self._apply_draw(pos)

    def _apply_draw(self, pos):
        """Apply the current draw mode to the clicked cell."""
        r, c = pos
        mode = self.draw_mode

        if mode == "start":
            # Move start to new position
            sr, sc = self.start
            self.grid[sr][sc] = "empty"
            self._color_cell(sr, sc, COLOR["empty"])
            self.start = (r, c)
            self.grid[r][c] = "start"
            self._color_cell(r, c, COLOR["start"])

        elif mode == "target":
            # Move target to new position
            tr, tc = self.target
            self.grid[tr][tc] = "empty"
            self._color_cell(tr, tc, COLOR["empty"])
            self.target = (r, c)
            self.grid[r][c] = "target"
            self._color_cell(r, c, COLOR["target"])

        elif mode == "wall":
            if self.grid[r][c] == "empty":
                self.grid[r][c] = "wall"
                self._color_cell(r, c, COLOR["wall"])

        elif mode == "erase":
            if self.grid[r][c] not in ("start", "target"):
                self.grid[r][c] = "empty"
                self._color_cell(r, c, COLOR["empty"])

    def _set_mode(self, mode):
        """Switch the current drawing mode."""
        self.draw_mode = mode
        self.mode_label.config(text=f"Mode: {mode.upper()}")

    # ──────────────────────────────────────────
    #  RESET GRID
    # ──────────────────────────────────────────
    def _reset(self):
        self.running   = False
        self.generator = None
        self.step_count = 0
        # Clear all cells
        for r in range(ROWS):
            for c in range(COLS):
                self.grid[r][c] = "empty"
        # Re-place start and target
        self.grid[self.start[0]][self.start[1]]   = "start"
        self.grid[self.target[0]][self.target[1]] = "target"
        self._draw_grid()
        self.status_var.set("Grid reset. Draw walls then press RUN.")
        self.stats_var.set("Steps: 0 | Explored: 0 | Path: -")
        print("[Grid] Reset complete.")

    # ──────────────────────────────────────────
    #  START SEARCH ANIMATION
    # ──────────────────────────────────────────
    def _start_search(self):
        if self.running:
            return

        algo_name = self.algo_var.get()
        self.current_algo = algo_name
        print(f"\n{'='*40}")
        print(f"  Starting {algo_name}")
        print(f"  Start: {self.start}  Target: {self.target}")
        print(f"{'='*40}")

        # Clear previous visualization (keep walls, start, target)
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] in ("explored", "frontier", "path"):
                    self.grid[r][c] = "empty"
                    self._color_cell(r, c, COLOR["empty"])

        # Re-color start and target after clear
        self._color_cell(self.start[0],  self.start[1],  COLOR["start"])
        self._color_cell(self.target[0], self.target[1], COLOR["target"])

        # Map algorithm name to generator function
        gen_map = {
            "BFS":           lambda: bfs(self.grid, self.start, self.target),
            "DFS":           lambda: dfs(self.grid, self.start, self.target),
            "UCS":           lambda: ucs(self.grid, self.start, self.target),
            "DLS":           lambda: dls(self.grid, self.start, self.target),
            "IDDFS":         lambda: iddfs(self.grid, self.start, self.target),
            "Bidirectional": lambda: bidirectional(self.grid, self.start, self.target),
        }
        self.generator  = gen_map[algo_name]()
        self.running    = True
        self.step_count = 0
        self.status_var.set(f"Running {algo_name}... watch it explore!")
        self._animate()

    # ──────────────────────────────────────────
    #  ANIMATION LOOP (called repeatedly via after())
    # ──────────────────────────────────────────
    def _animate(self):
        if not self.running or self.generator is None:
            return

        try:
            result = next(self.generator)
        except StopIteration:
            self.running = False
            self.status_var.set("Search complete!")
            print("[Animation] Search complete.")
            return

        kind = result[0]

        if kind == "step":
            _, explored, frontier = result
            self.step_count += 1

            # Color explored cells (light blue)
            for (r, c) in explored:
                if self.grid[r][c] == "empty":
                    self.grid[r][c] = "explored"
                    self._color_cell(r, c, COLOR["explored"])

            # Color frontier cells (orange) – waiting to be explored
            for (r, c) in frontier:
                if self.grid[r][c] == "empty":
                    self._color_cell(r, c, COLOR["frontier"])

            # Always keep start and target visible on top
            self._color_cell(self.start[0],  self.start[1],  COLOR["start"])
            self._color_cell(self.target[0], self.target[1], COLOR["target"])

            self.stats_var.set(
                f"Steps: {self.step_count} | "
                f"Explored: {len(explored)} | "
                f"Frontier: {len(frontier)}"
            )

            # Schedule the next animation step after DELAY ms
            self.root.after(DELAY, self._animate)

        elif kind == "path":
            path = result[1]
            self.running = False

            if path:
                # Animate path drawing cell by cell
                for i, (r, c) in enumerate(path):
                    self.root.after(i * 30, lambda r=r, c=c: (
                        self._color_cell(r, c, COLOR["path"])
                    ))
                # Keep start and target colored on top after path drawn
                self.root.after(len(path) * 30 + 10, lambda: (
                    self._color_cell(self.start[0],  self.start[1],  COLOR["start"]),
                    self._color_cell(self.target[0], self.target[1], COLOR["target"])
                ))
                self.status_var.set(
                    f"{self.current_algo} found path! "
                    f"Length = {len(path)} | Steps = {self.step_count}"
                )
                self.stats_var.set(
                    f"Steps: {self.step_count} | "
                    f"Explored: {self.step_count} | "
                    f"Path Length: {len(path)}"
                )
                print(f"[Result] Path visualized. Length = {len(path)}")
            else:
                self.status_var.set(
                    f"{self.current_algo}: No path found! (blocked by walls)"
                )
                print("[Result] No path found.")


# =================================================================
#  ENTRY POINT
# =================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = AIPathfinderApp(root)
    root.mainloop()