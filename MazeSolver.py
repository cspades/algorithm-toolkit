import os
import time
import numpy as np
from algorithm_library import DisjointEnsemble

class MazeSolver:
    """
    Maze-solving algorithm implementation via FloodFill with basic GUI.
    Given a maze and initial position, simulates the iterations of the
    FloodFill algorithm, and provides tools to generate random mazes.

    To setup the maze, form a 2-D array of unsigned integers
    such that the binary representation of the coordinate describes
    the structure of the maze at that position, where redundant
    placements are deduplicated with OR logic:

        0001 = 1 = North Wall (Up)
        0010 = 2 = South Wall (Down)
        0100 = 4 = West Wall (Left)
        1000 = 8 = East Wall (Right)

    For instance, the sequence of numbers (11, 7, 10) will represent the
    following adjacent maze cells in the (y,x)-coordinate system:

    . ------> x
    | --- ---
    | 11 | 7  10 |
    v --- --- ---
    y
    """
    class Coordinate:
        def __init__(self, pos: tuple[int, int] = (0,0), dir: str = None):
            self.y = pos[0]
            self.x = pos[1]
            self.dir = dir
        def getPos(self):
            return (self.y, self.x)
        def setPos(self, pos: tuple[int, int]):
            self.y = pos[0]
            self.x = pos[1]
        def getDir(self):
            return self.dir
        def setDir(self, dir: str):
            self.dir = dir

    class Direction:
        UP = "u"
        DOWN = "d"
        LEFT = "l"
        RIGHT = "r"

    class MazeObject:
        """
        Constant class attributes describing maze objects
        and mouse orientations along with their unique
        visual representation in the GUI.
        """
        V_WALL = ":"
        V_WALL_OBS = "|"
        V_OPEN = " "
        H_WALL = "---"
        H_WALL_OBS = "==="
        H_OPEN = "   "
        MOUSE_UP = " ^ "
        MOUSE_LEFT = " < "
        MOUSE_RIGHT = " > "
        MOUSE_DOWN = " v "
        MOUSE_ORIENT = {
            "u": MOUSE_UP,
            "d": MOUSE_DOWN,
            "l": MOUSE_LEFT,
            "r": MOUSE_RIGHT
        }
    
    # Directional Encoding Dictionary
    bitDirection = {
        "u": 1, # Up
        "d": 2, # Down
        "l": 4, # Left
        "r": 8, # Right
        "o": 0, # Open (None)
        "c": 15 # Closed (All)
    }

    # Map from coordinate delta to relative directions.
    deltaDir = {
        (0,1): {"cur": "r", "adj": "l"},
        (0,-1): {"cur": "l", "adj": "r"},
        (1,0): {"cur": "d", "adj": "u"},
        (-1,0): {"cur": "u", "adj": "d"}
    }

    # Map from direction to coordinate delta.
    dirDelta = {
        "u": (-1,0),
        "d": (1,0),
        "l": (0,-1),
        "r": (0,1)
    }

    def __init__(self, maze: np.ndarray, mouse: tuple[int, int] = (0,0)):
        """
        Initialize the maze from a NumPy array encoding of the maze.
        """
        # Validate input parameters.
        if any([
            maze.shape[0] < 1,
            maze.shape[1] < 1,
            mouse[0] < 0,
            mouse[0] >= maze.shape[0],
            mouse[1] < 0,
            mouse[1] >= maze.shape[1],
        ]):
            raise ValueError(
                f"[MazeSolver] Maze dimensions should be positive, and the mouse initial position must be located in the maze.\n" +
                f"Maze Dimensions: ({maze.shape[0]}, {maze.shape[1]})\n" +
                f"Mouse Coordinates: ({mouse[0]}, {mouse[1]})"
            )
        # Initialize maze variables. Persist maze state for efficient further solving.
        self.mouse = self.Coordinate(mouse, self.Direction.DOWN)
        self.maze = maze.astype(int)
        self.observedMaze = np.zeros(self.maze.shape, dtype=int)
        self.floodFillMatrix = np.zeros(self.maze.shape, dtype=int)

    def __repr__(self):
        return self._drawMaze(self.maze, mouse=self.mouse)
    
    def solve(self, dest: tuple[int, int] = (0,0), sim: int = -1, history: bool = False):
        """
        Solve the maze by relocating the mouse to the destination coordinates (y,x).
        :param
        """
        # Unpack.
        y, x = dest
        # Validate destination.
        if any([
            y < 0,
            y >= self.maze.shape[0],
            x < 0,
            x >= self.maze.shape[1],
        ]):
            raise ValueError(
                f"[MazeSolver] Destination coordinates must be located in the maze.\n" +
                f"Maze Dimensions: ({self.maze.shape[0]}, {self.maze.shape[1]})\n" +
                f"Destination Coordinates: ({y}, {x})"
            )
        
        # Reset distances for FloodFill.
        self.floodFillMatrix = np.zeros(self.maze.shape, dtype=int)

        # Solve the maze step-by-step.
        while True:

            # Observe walls to update observedMaze.
            for delta, wallDirs in self.deltaDir.items():
                if self._checkWall(self.mouse.getPos(), self.maze, wallDirs["cur"]):
                    # Add wall for current cell and adjacent cell.
                    adjCell = self._applyDelta(self.mouse.getPos(), delta)
                    self._addWall(self.mouse.getPos(), self.observedMaze, wallDirs["cur"])
                    try:
                        self._addWall(adjCell, self.observedMaze, wallDirs["adj"])
                    except IndexError:
                        # Wall placement exceeds maze boundary. Do nothing.
                        pass

            # Print maze state with non-observed vs. observed walls.
            if not history:
                # Clear terminal screen. 
                os.system('cls' if os.name == 'nt' else 'clear')
            print(self._drawMaze(self.maze, self.mouse, self.observedMaze))

            # Terminate solver if the destination is reached.
            if self.mouse.getPos() == dest:
                # Terminate.
                print(f"Destination reached! Saving and terminating the solver...")
                break

            # Apply solver simulation mode.
            if sim > 0:
                # Wait for 2 seconds and automatically progress.
                time.sleep(sim)
            else:
                # Step Mode: User input progresses the solver.
                userInput = input(f"> Continue progress on MazeSolver? [y/n]\nAnswer: ")
                if userInput == "n":
                    # Terminate.
                    print(f"Solver state saved. Terminating the solver...")
                    break
            
            # Solve the (observed) maze via FloodFill.
            self.floodfill(dest)
                
            # Compute optimal direction to solve the maze.
            minDist = float('inf')
            minDelta = set()
            for delta, wallDirs in self.deltaDir.items():
                # Validate direction.
                adjCell = self._applyDelta(self.mouse.getPos(), delta)
                if not self._checkMazeBounds(adjCell, self.observedMaze):
                    # Direction exceeds maze boundary. Skip.
                    continue
                # Identify the shortest path to destination.
                adjDist = self._posLookup(self.floodFillMatrix, adjCell)
                if not self._checkWall(self.mouse.getPos(), self.observedMaze, wallDirs["cur"]) \
                    and adjDist <= min(self._posLookup(self.floodFillMatrix, self.mouse.getPos()), minDist):
                    # Update minimum and direction cell.
                    minDist = adjDist
                    minDelta.add(delta)
            if not minDelta:
                # No viable direction. Terminate.
                print(f"Maze is unsolvable! Terminating...")
                break

            # If multiple directions are viable, prioritize moving in the direction of the destination.
            # To sort in order of most aligned to least aligned directions, compute the inner product.
            destDelta = self._computeDelta(dest, self.mouse.getPos())
            bestDelta = sorted(
                [(int(np.inner(delta, destDelta)), delta) for delta in self.deltaDir if delta in minDelta],
                key=lambda x : x[0],
                reverse=True
            )[0][1]

            # Move mouse. If not facing in direction of movement, change direction.
            mvCell = self._applyDelta(self.mouse.getPos(), bestDelta)
            if self.mouse.getDir() != self.deltaDir[bestDelta]["cur"]:
                # Change direction.
                self.mouse.setDir(self.deltaDir[bestDelta]["cur"])
            else:
                # Move in optimal direction.
                self.mouse.setPos(mvCell)

    def floodfill(self, dest: tuple[int, int]):
        """
        Execute the FloodFill algorithm to update all
        Manhattan distances in the (observed) maze
        to represent the distance to dest.
        """
        # Instantiate FloodFill parameters.
        # Set destination distance to 0.
        cellStack = [self.mouse.getPos()]
        cellSet = set([self.mouse.getPos()])
        self.floodFillMatrix[dest[0], dest[1]] = 0

        # FloodFill
        while cellStack:
            # Pop stack.
            cell = cellStack.pop()
            cellSet.remove(cell)

            # If cell is the destination, do NOT update the distance.
            # The distance is a constant at 0.
            if cell == dest:
                continue

            # Compute minimum adjacent distance.
            minDist = float('inf')
            for delta, wallDirs in self.deltaDir.items():
                # Reachable?
                adjCell = self._applyDelta(cell, delta)
                if self._checkWall(cell, self.observedMaze, wallDirs["cur"]) or not self._checkMazeBounds(adjCell, self.observedMaze):
                    # Unreachable cell.
                    continue
                # Identify adjacent FloodFill distance.
                adjDist = self._posLookup(self.floodFillMatrix, adjCell)
                if adjDist < minDist:
                    # Update minimum distance.
                    minDist = adjDist
            
            
            # Set current cell Manhattan distance to minimum distance of adjacent reachable cells.
            if self._posLookup(self.floodFillMatrix, cell) == minDist + 1 or minDist == float('inf'):
                # No update necessary if there are no reachable cells or the distance is set to minDist + 1.
                continue
            else:
                # Update.
                self._posSet(self.floodFillMatrix, cell, minDist + 1)

            # Push adjacent reachable cells into the stack.
            for delta, wallDirs in self.deltaDir.items():
                adjCell = self._applyDelta(cell, delta)
                if all([
                    adjCell not in cellSet,
                    not self._checkWall(cell, self.observedMaze, wallDirs["cur"]),
                    self._checkMazeBounds(adjCell, self.observedMaze) 
                ]):
                    cellStack.append(adjCell)
                    cellSet.add(adjCell)
    
    @classmethod
    def _checkWall(cls, pos: tuple[int, int], maze: np.ndarray, dir: str = "c"):
        return bool(maze[pos[0], pos[1]] & cls.bitDirection.get(dir, 15))
    
    @classmethod
    def _addWall(cls, pos: tuple[int, int], maze: np.ndarray, dir: str = "o"):
        maze[pos[0], pos[1]] |= cls.bitDirection.get(dir, 0)

    @classmethod
    def _deleteWall(cls, pos: tuple[int, int], maze: np.ndarray, dir: str = "o"):
        maze[pos[0], pos[1]] &= ~cls.bitDirection.get(dir, 0)
    
    @staticmethod  
    def _checkMazeBounds(pos: tuple[int, int], maze: np.ndarray):
        # Check if the coordinates of pos are within the bounds of maze.
        return pos[0] >= 0 and pos[0] < maze.shape[0] and pos[1] >= 0 and pos[1] < maze.shape[1]
    
    @staticmethod
    def _computeDelta(a: tuple[int, int], b: tuple[int, int]):
        return tuple(np.subtract(a, b))

    @staticmethod
    def _applyDelta(pos: tuple[int, int], delta: tuple[int, int]):
        return tuple(np.add(pos, delta))
    
    @staticmethod
    def _posLookup(matrix: np.ndarray, pos: tuple[int, int]):
        return matrix[pos[0], pos[1]]
    
    @staticmethod
    def _posSet(matrix: np.ndarray, pos: tuple[int, int], value):
        matrix[pos[0], pos[1]] = value

    @staticmethod
    def _maze_2d_to_2d(pos: tuple[int, int]):
        return 2*pos[0]+1, 2*pos[1]+1
    
    @staticmethod
    def _drawMaze(maze: np.ndarray, mouse = None, observedMaze: np.ndarray = None):
        """
        Utilizing the input and observed mazes, expand the coordinate
        system to (2X-1,2Y-1) to draw the current state of the maze.
        Observed and non-observed objects are depicted differently,
        e.g. an observed horizontal wall appears as '===' but an
        unobserved horizontal wall appears as '---'.
        """
        # Maze Parameters
        Y, X = maze.shape[0], maze.shape[1]
        # Visualize maze as 2-D ndarray of String.
        WALL_GAP = " "
        CELL_SPACE = " " * 3
        graphicMaze = np.full(((2*Y+1), (2*X+1)), '?', dtype="U3")
        for y in range(Y):
            for x in range(X):
                # Map to raw maze coordinates.
                j, i = MazeSolver._maze_2d_to_2d((y, x))
                """
                Maze Cells
                """
                if mouse is not None and mouse.getPos() == (y,x):
                    # Set mouse.
                    graphicMaze[j,i] = MazeSolver.MazeObject.MOUSE_ORIENT[mouse.getDir()]
                else:
                    # Empty cell.
                    graphicMaze[j,i] = CELL_SPACE
                """
                Maze Walls
                """
                if MazeSolver._checkWall((y,x), maze, "u"):
                    # North Wall
                    northWall = MazeSolver.MazeObject.H_WALL_OBS if observedMaze is not None and MazeSolver._checkWall((y,x), observedMaze, "u") else MazeSolver.MazeObject.H_WALL
                    graphicMaze[j-1,i] = northWall
                elif graphicMaze[j-1,i]== "?":
                    graphicMaze[j-1,i] = MazeSolver.MazeObject.H_OPEN
                if MazeSolver._checkWall((y,x), maze, "d"):
                    # South Wall
                    southWall = MazeSolver.MazeObject.H_WALL_OBS if observedMaze is not None and MazeSolver._checkWall((y,x), observedMaze, "d") else MazeSolver.MazeObject.H_WALL
                    graphicMaze[j+1,i] = southWall
                elif graphicMaze[j+1,i] == "?":
                    graphicMaze[j+1,i] = MazeSolver.MazeObject.H_OPEN
                if MazeSolver._checkWall((y,x), maze, "l"):
                    # West Wall
                    westWall = MazeSolver.MazeObject.V_WALL_OBS if observedMaze is not None and MazeSolver._checkWall((y,x), observedMaze, "l") else MazeSolver.MazeObject.V_WALL
                    graphicMaze[j,i-1] = westWall
                elif graphicMaze[j,i-1] == "?":
                    graphicMaze[j,i-1] = MazeSolver.MazeObject.V_OPEN
                if MazeSolver._checkWall((y,x), maze, "r"):
                    # East Wall
                    eastWall = MazeSolver.MazeObject.V_WALL_OBS if observedMaze is not None and MazeSolver._checkWall((y,x), observedMaze, "r") else MazeSolver.MazeObject.V_WALL
                    graphicMaze[j,i+1] = eastWall
                elif graphicMaze[j,i+1] == "?":
                    graphicMaze[j,i+1] = MazeSolver.MazeObject.V_OPEN
                    
        # Print maze.
        outputMaze = ""
        for j in range(2*Y+1):
            for i in range(2*X+1):
                # Fill in wall gaps.
                if j % 2 == 0 and i % 2 == 0:
                    graphicMaze[j,i] = WALL_GAP
                # Append to graphic.
                outputMaze += graphicMaze[j,i]
            outputMaze += "\n"
        return outputMaze
    
    @classmethod
    def generateMaze(cls, length: int, width: int, seed: int = None):
        """
        Generate a random bounded maze using Kruskal's Algorithm.
        """
        # Initialize maze.
        randomMaze = np.full((length, width), 15)
        # Randomly grow a set of maze coordinates
        # that are connected via destroying walls.
        rng = np.random.default_rng(seed)
        disjointCells = DisjointEnsemble((j,i) for j in range(length) for i in range(width))
        while True:
            # Unpack a random starting point.
            j, i = rng.integers(low=0, high=length), rng.integers(low=0, high=width)
            # Randomly choose a neighbor not in the SCC
            # of the coordinate and within bounds.
            adjCoordinates = [
                x for x in [(j,i+1), (j,i-1), (j+1,i), (j-1,i)]
                if all([
                    cls._checkMazeBounds(x, randomMaze),
                    disjointCells.findTreeRoot(x) != disjointCells.findTreeRoot((j,i))
                ])
            ]
            rng.shuffle(adjCoordinates)
            if not adjCoordinates:
                # No candidates for connection. Randomly choose another coordinate.
                continue
            else:
                # Connect with adjacent cell.
                adjCell = adjCoordinates[0]
                # Compute differential.
                delta = cls._computeDelta(adjCell, (j,i))
                # Destroy both walls.
                cls._deleteWall((j, i), randomMaze, cls.deltaDir[delta]["cur"])
                cls._deleteWall(adjCell, randomMaze, cls.deltaDir[delta]["adj"])
                # Merge their sets.
                disjointCells.treeUnion((j,i), adjCell)
                union = disjointCells.getSet((j,i))
                if len(union) == length * width:
                    # Connected maze. Terminate.
                    break
        # Return randomized maze and mouse.
        return randomMaze, (rng.integers(low=0, high=length), rng.integers(low=0, high=width))
    
print(f"Testing MazeSolver...")
# Generate random maze and mouse position.
inputMaze, mouse = MazeSolver.generateMaze(8,40)
# Instantiate MazeSolver.
mazeSolver = MazeSolver(maze=inputMaze, mouse=mouse)
# Path Planning
mazeSolver.solve((0, 0), sim=0.05, history=False)
mazeSolver.solve((7, 39), sim=0.05, history=False)
mazeSolver.solve((4, 20), sim=0.05, history=False)