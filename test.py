from algorithm_library import *
from ml_library import *
import time

""" Algorithm Library Testing """

# print(f"Testing HeapCache...")
# heap = HeapCache(reverse=False, capacity=4)
# print(f"Empty Heap: ", heap)
# heap.insertData("a", "a", 10)
# heap.insertData("b", "b", 1)
# heap.insertData("c", "c", 12)
# heap.insertData("d", "d", 30)
# heap.insertData("e", "e", -3)
# print(f"Insertion + Overflow Test: ", heap)
# heap.popData("c")
# heap.popData("e")
# print(f"Deletion Test: ", heap)
# print(f"HeapSort Test: ", heap.__repr__(False))
# testCache = [
#     HeapCache.Data("a", "a", 10, 0),
#     HeapCache.Data("b", "b", 1, 1),
#     HeapCache.Data("c", "c", 12, 2),
#     HeapCache.Data("d", "d", 30, 3),
#     HeapCache.Data("e", "e", -3, 4)
# ]
# heap.heapify(cache=testCache)
# print(f"Heapify Test: " + str([f"({x.getData()}, {x.getOrder()}, {x.getIndex()})" for x in testCache]))

# print(f"Testing QuickSort...")
# a = [12,1,-20,50,12,12,12,13481,3,-1324,5]
# b = []
# QuickSort.sort(a, reverse=True)
# print(a)
# QuickSort.sort(a, reverse=False)
# print(a)
# QuickSort.sort(b, reverse=False)
# print(b)

# print(f"Testing LinkedList...")
# ll = LinkedList()
# ll.append(LinkedList.Node("1"))
# ll.append(LinkedList.Node("2"))
# ll.append(LinkedList.Node("3"))
# ll.append(LinkedList.Node("4"))
# ll.append(LinkedList.Node("5"))
# print(ll)
# ll.delete("3")
# print(ll)
# ll.swap("5", "2")
# print (ll)
# ll.reverse()
# print(ll)
# ll.clear()
# print(ll)
# ll.reverse()
# print(ll)

# print(f"Testing FibonacciCache...")
# fc = FibonacciCache()
# t1 = time.time()
# fc.fibonacci(2500)
# t2 = time.time()
# print(f"{fc} | Time Spent: {t2-t1} sec")
# t3 = time.time()
# fc.fibonacci(5000)
# t4 = time.time()
# print(f"{fc} | Time Spent: {t4-t3} sec")
# recNum = 25
# t5 = time.time()
# fc.fibonacci(recNum, False)
# t6 = time.time()
# print(f"Fibonacci Number {recNum}: {fc.fiboCache[recNum]} | Recursive Time Spent: {t6-t5} sec")

# print(f"Testing TarjanSCC()...")
# G = [
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1],
#     [0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0],
#     [1, 0, 0, 1, 1],
# ]
# t = TarjanSCC(G)
# scc = t.tarjan_dfs(reverse=False)
# print(scc)
        
# print(f"Testing DijkstraBFS()...")
# G = [
#     [0, 1, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 5, 0],
#     [0, 0, 1, 0, 1],
#     [0, 0, -2, -4, 0],
# ]
# d = DijkstraBFS(G, maximal=False)
# dist, path = d.bfs()
# print(dist)
# print(path)

# print(f"Testing KruskalMST()...")
# G = [
#     [0, 1, -1, 1, 0],
#     [1, 0, 1, 1, 2],
#     [4, 2, 0, 5, 0],
#     [1, -1, 1, 0, 0],
#     [7, 0.5, -2, -4, 0],
# ]
# d = KruscalMST(G, maximal=True)
# tree, score = d.mst()
# print(tree)
# print(score)

# print(f"Testing KnapSack()...")
# VALUE = [4, 2, 1, 6, 7]
# COST = [2, 2, 3, 4, 5]
# ks = KnapSack(VALUE, COST, weight=8, repetition=False)
# print(ks.compute_knapsack())

# print(f"Testing LevenshteinDP()...")
# ld = LevenshteinDP("hello", "brushllw")
# print(ld.edit_distance())

# print(f"Testing WaterCapture()...")
# BARS = [ 4, 2, 1, 3, 0, 1 ]
# wc = WaterCapture(BARS)
# print(wc.water_volume())
# print(wc.water_volume_alt())

# print(f"Testing Numerics...")
# N = 100
# K = 10
# S = 100000
# count = 0
# for _ in range(S):
#     x = Numerics.randNFromRandK(n=N, k=K)
#     if x == 0:
#         count += 1
# print(f"Expected Probability: {1 / N} | Experimental Ratio: {count / S}")
# # Sample
# print(Numerics.randomSample(N, K))
# # Shuffle
# print(Numerics.randomSample(N, N))

""" Machine Learning Library Testing """

# print(f"Testing MatrixFactorize()...")
# SEED = 0
# SHAPE = (5,9)
# rng = np.random.default_rng(SEED)
# M = rng.integers(100, size=SHAPE)
# Z = 3
# m = MatrixFactorize(
#     M,
#     Z,
#     mask=rng.integers(2, size=SHAPE),
#     bias=True,
#     seed=SEED
# )
# m.fit()
# print(m)

# print(f"Testing FactorizationMachine()...")
# SEED = 0
# N = 1000
# X = 10
# T = 1
# H = 10
# VALUES = (-100, 100)
# rng = np.random.default_rng(SEED)
# INPUT = (VALUES[1] - VALUES[0]) * rng.random(size=(N,X)) + VALUES[0]
# TARGET = rng.integers(2, size=(N,T))
# fm = FactorizationMachine(
#     X, H, seed=SEED
# )
# fm.fit(
#     INPUT,
#     TARGET,
#     mask=rng.integers(2, size=(N,X)),
#     cycles=100,
#     lr=2e-3,
#     batch_frac=0.01,
#     regularize=1e-2,
#     patience=5,
#     verbose=True
# )