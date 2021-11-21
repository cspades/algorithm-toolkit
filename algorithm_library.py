"""
Algorithm Library (Python v3.9.6+)
Implemented by Cory Ye
For personal educational review.
"""
import sys
import heapq


class TarjanSCC():
    """
    Compute all strongly-connected components in a directed graph G.
    Utilizes Tarjan's strongly-connected components recursion DFS algorithm.
    Returns a list of strongly-connected components.
    """

    def __init__(self, graph):
        """
        Instantiate graph information for strongly-connected component searching of G.
        :param graph <list<list>>:  Adjacency matrix for the graph G. Nodes are indexed by 
                                    non-negative integers, i.e. 0, 1, 2, ...
        """

        self.G = graph                  # Adjacency Matrix for Graph
        self.dfs_stack = []             # DFS Stack
        self.index = 0                  # Exploration Index
        self.D = {
            k: {
                'index': None,          # Track exploration index.
                'minlink': None,        # Track minimal sub-tree / reachable index.
                'instack': False        # Track DFS stack presence (for efficient lookup).
            } 
            for k in range(len(graph))
        }

    def tarjan_dfs(self, reverse=False):
        """
        Execute Tarjan's strongly-connected components algorithm. Sorted in topological order from source to sink.
        :param reverse <bool>:  Topological sort on list of SCC from sinks to sources instead of sources to sinks.
        """

        # Search for strongly-connected components for all nodes in the graph.
        SCC = []
        for v in range(len(self.G)):
            # Skip explored nodes.
            if self.D[v]['index'] is None:
                # Identify strongly-connected components associated with minimal reachable node v.
                component = self.scc(v)
                if component:
                    SCC.append(component)

        # Topological Sort
        if not reverse:
            # Reverse the discovered list of SCC to sort 
            # in order from sources to sinks instead of 
            # sinks to sources in the graph G.
            SCC.reverse()
        
        # Output list of SCC.
        return SCC

    def scc(self, v):
        """
        Identify strongly-connected components associated with the minimal reachable node v.
        """
        # Process the node v. Set the exploration index, 
        # initialize the minlink index, and push into stack.
        self.D[v]['index'] = self.index
        self.D[v]['minlink'] = self.index
        self.index += 1
        self.dfs_stack.append(v)
        self.D[v]['instack'] = True
        
        # Explore adjacent nodes.
        for w in range(len(self.G)):
            # Adjacent reachable nodes.
            if self.G[v][w] != 0:
                # Unexplored node.
                if self.D[w]['index'] is None:
                    # Analyze strongly-connected sub-component of node w.
                    self.scc(w)
                    # Update the minimum exploration index reachable from w.
                    self.D[v]['minlink'] = min(
                        self.D[v]['minlink'],
                        self.D[w]['minlink']
                    )
                # Explored node in the DFS stack.
                elif self.D[w]['instack']:
                    # Update the minimum exploration index relative to
                    # the back-edge node. Do NOT utilize the minimum 
                    # reachable exploration index of the back-edge node, 
                    # which considers minimum reachable exploration indices 
                    # of the sub-tree of the back-edge node!
                    self.D[v]['minlink'] = min(
                        self.D[v]['minlink'],
                        self.D[w]['index']
                    )
                # Explored nodes not in the DFS stack are pre-discovered SCC's.
                else:
                    # Neglect irrelevant nodes.
                    continue
        
        # Output the SCC if the node is a minimal reachable node of the SCC.
        scc_detect = []
        if self.D[v]['minlink'] == self.D[v]['index']:
            # Include nodes in the sub-tree of the minimal reachable node.
            while self.dfs_stack and self.D[self.dfs_stack[-1]]['index'] >= self.D[v]['index']:
                w = self.dfs_stack.pop()
                scc_detect.append(w)
                self.D[w]['instack'] = False

        return scc_detect


class DijkstraBFS():

    def __init__(self, graph, maximal=False):
        """
        Instantiate graph information for minimal breadth-first searching in Dijkstra's Algorithm.
        :param graph <list<list>>:  Adjacency matrix (with optional weights) for the graph G. 
                                    Nodes are indexed by non-negative integers, i.e. 0, 1, 2, ...
        :param maximal <bool>:      Return maximal path(s) / distance(s) instead.
        """
        
        self.G = graph
        extrema = float('inf') if not maximal else -float('inf')
        self.dist = {
            x: {
                y: extrema if x != y else 0
                for y in range(len(graph))
            } for x in range(len(graph))
        }
        self.path = {
            x: {
                y: [] if x != y else [x]
                for y in range(len(graph))
            } for x in range(len(graph))
        }
        self.maximal = maximal

    def bfs(self, initial_node=None):
        """
        Perform a minimal (or maximal) breadth-first search of the graph G.
        :param initial_node <int>:  Initial node specification instead of processing entire graph.
        """

        # Search from all initial nodes in case of directed or disconnected components.
        task = list(range(len(self.G)))
        if initial_node is not None and initial_node in task:
            task = [initial_node]
        for v in task:
            # Reset queue and processed set.
            heap = []
            heapq.heappush(
                heap, 
                (0,v)
            )
            processed = set()
            # BFS
            while heap:
                # Pop minimal node. Pre-emptively set node as processed.
                _, a = heapq.heappop(heap)
                processed.add(a)

                # Search for adjacent nodes.
                for b in range(len(self.G)):
                    if b != a and self.G[a][b] != 0:

                        # Update distance and path.
                        if any([
                            not self.maximal and self.dist[v][b] > self.dist[v][a] + self.G[a][b],
                            self.maximal and self.dist[v][b] < self.dist[v][a] + self.G[a][b]
                        ]):
                            self.dist[v][b] = self.dist[v][a] + self.G[a][b]
                            self.path[v][b] = self.path[v][a] + [b]

                        # Push un-processed adjacent nodes onto priority heap / queue.
                        if b not in processed:
                            heapq.heappush(
                                heap,
                                (self.G[a][b], b) if not self.maximal else (-self.G[a][b], b)
                            )

        # Output distance(s) and path(s) in the graph G.
        return self.dist, self.path


class KruscalMST():

    def __init__(self, graph, maximal=False):
        """
        Instantiate graph information for Kruskal's Minimal Spanning Tree algorithm.
        :param graph <list<list>>:  Adjacency matrix (with optional weights) for the graph G. 
                                    Nodes are indexed by non-negative integers, i.e. 0, 1, 2, ...
        :param maximal <bool>:      Return a maximal spanning tree instead.
        """

        # Instantiate graph and sort edge weights.
        self.G = graph
        self.E = []
        for i in range(len(graph)):
            for j in range(len(graph)):
                # Insert weighted edge into priority heap / queue.
                if graph[i][j] != 0:    # Non-existent edge.
                    heapq.heappush(
                        self.E,
                        (graph[i][j], (i,j)) if not maximal else (-graph[i][j], (i,j))
                    )
        self.setcache = {
            x: set([x]) for x in range(len(graph))
        }
        self.maximal = maximal

    def mst(self):
        """
        Compute a list of edges that constitutes the minimal spanning tree of the graph G.
        Return list of edges constituting minimal spanning tree, and the cumulative tree edge weight score.
        """

        # Build minimal spanning tree.
        tree = []
        score = 0
        while len(tree) < len(self.G):

            # Pop the minimal edge.
            w, e = heapq.heappop(self.E)

            # Combine sets.
            if self.setcache[e[0]] != self.setcache[e[1]]:

                # Union.
                u = self.setcache[e[0]] | self.setcache[e[1]]
                self.setcache[e[0]] = u
                self.setcache[e[1]] = u

                # Append edge to MST.
                tree.append(e)
                if not self.maximal:
                    score += w
                else:
                    score -= w

        return tree, score


class KnapSack():

    def __init__(self, value, cost, weight=None, repetition=False):
        """
        Instantiate dynamic memory for the KnapSack Problem.
        :param value <list<float>>:     List of values / gains / profits for items in the knapsack.
        :param cost <list<int>>:        List of (positive integer) weights / losses / costs for items in the knapsack.
        :param weight <int|None>:       Maximum weight of knapsack. If not set, default to sum of all costs.
        :param repetition <bool>:       Repeat items in knapsack.
        """

        # Validate input.
        if any([
            len(value) != len(cost),
            any(not isinstance(x, int) or x <= 0 for x in cost),
            weight is not None and not isinstance(weight, int)
        ]):
            print(
                f"""[KnapSackError] Cannot solve knapsack problem with non-integral or non-positive weight(s) / cost(s).
                    For non-integral cost(s), either approximate costs to nearest integer or utilize linear programming (LP) 
                    optimization algorithms instead.""",
                file=sys.stderr,
                flush=True
            )
            sys.exit(1)

        # Instantiate dynamic memory.
        self.value = value
        self.cost = cost
        self.limit = sum(cost)
        if weight is not None:
            # Set custom knapsack limit for efficiency.
            self.limit = int(weight)
        self.Q = {  # Reward matrix of shape (weight, item).
            **{ w: { -1: (0, []) } for w in range(self.limit+1) },
            **{ 0: { k: (0, []) for k in range(-1, len(value)) } }
        }
        self.rep = repetition

    def compute_knapsack(self):
        """
        Compute the optimal knapsack via dynamic programming.
        """

        Q_opt = (-float('inf'), [])
        for w in range(self.limit+1):
            for k in range(len(self.value)): 
                if self.cost[k] > w:
                    # Cannot add item into knapsack without overflowing the limit.
                    # Set to knapsack not including item k.
                    self.Q[w][k if not self.rep else -1] = self.Q[w][k-1 if not self.rep else -1]
                else:
                    test_val = self.Q[w-self.cost[k]][k-1 if not self.rep else -1][0] + self.value[k]
                    if test_val > self.Q[w][k-1 if not self.rep else -1][0]:
                        # Include new item. Update knapsack.
                        self.Q[w][k if not self.rep else -1] = (
                            test_val,
                            self.Q[w-self.cost[k]][k-1 if not self.rep else -1][1] + [k]
                        )
                    else:
                        # Exclude new item.
                        self.Q[w][k if not self.rep else -1] = self.Q[w][k-1 if not self.rep else -1]
                # Update optimal knapsack.
                if self.Q[w][k if not self.rep else -1][0] > Q_opt[0]:
                    Q_opt = self.Q[w][k if not self.rep else -1]

        return Q_opt


class LevenshteinDP():

    def __init__(self, a, b):
        """
        Instantiate memory for computing edit distance between word a <str> and word b <str>.
        """

        # Store word strings.
        self.w1 = a
        self.w2 = b

        # Compute edit distance matrix with initial edit metrics for total deletion or insertion.
        self.edit = [
            [ max(i,j) if j == 0 or i == 0 else 0 for j in range(len(b)+1) ] 
            for i in range(len(a)+1)
        ]

    def edit_distance(self):
        """
        Compute the Levenshtein edit distance between the specified words.
        """

        # Loop through both words.
        for i in range(1, len(self.w1)+1):
            for j in range(1, len(self.w2)+1):

                # Edit. Test and penalize insert, delete, and replace.
                edit_penalty = 1 if self.w1[i-1] != self.w2[j-1] else 0
                self.edit[i][j] = min(
                    self.edit[i-1][j],
                    self.edit[i][j-1],
                    self.edit[i-1][j-1]
                ) + edit_penalty

        # Print optimal alignment score.
        return self.edit[-1][-1]


class WaterCapture():
    """
    [Greedy Water Capture Problem]
    Given a collection of wall(s) / barrier(s) with various height, 
    compute the optimal rectangular container represented by precisely 
    2 wall(s) / barrier(s) that capture the most rainwater measured by 
    the cross-sectional area of the container.

    |~~~~~~~~~~~~~~         
    | Water_Area=9 |
    |    |         |    |
    |____|____|____|____|____|
    """

    def __init__(self, height_vector):
        """
        Initialize parameters to solve the greedy water capture problem 
        given a vector of wall heights.
        :param height_vector <list<float>>: List of wall heights.
        """
        
        # Store container information.
        self.bars = height_vector

        # Instantiate search pointers.
        self.L = 0
        self.R = len(height_vector)-1
        self.x_area = 0

    def water_volume(self):
        """
        Compute the optimal volume of water captured by 
        a container formed from the wall(s) / barrier(s).
        """

        # Loop over all width(s) of all possible containers
        # from largest (len(self.bars) - 1) to smallest (1).
        for width in range(len(self.bars)-1, 0, -1):

            # Implement a greedy search method. Minimum bar height
            # of a container dictates the cross-sectional area,
            # so searching for a different bar on the side of
            # the container with greater height can only decrease
            # the cross-sectional area via decreasing the width
            # of the tested container.
            if self.bars[self.L] < self.bars[self.R]:
                # Track cross-sectional area.
                self.x_area = max(self.x_area, self.bars[self.L] * width)
                # Test different container.
                self.L += 1
            else:
                # Track cross-sectional area.
                self.x_area = max(self.x_area, self.bars[self.R] * width)
                # Test different container.
                self.R -= 1

        return self.x_area