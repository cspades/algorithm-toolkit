"""
Algorithm Library (Python v3.12.2+)
Implemented by Cory Ye
For personal educational review.
"""
import heapq
import math
import random
import sys
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, MutableSequence, Hashable
from typing import Any, TypeVar, Iterable

class TrieTagger:
    """
    Trie-based entity tagger to tag all instances of a library
    of string entities in a provided document string.

    Suppose N is the token length of a document, M is the size of the
    library of entities, and K is the token length of the longest
    entity in the library. Then the insertion time complexity is O(MK)
    from iteratively adding M entities with K length into the Trie,
    and the matching time complexity is O(NK) from iteratively searching
    for K length entities starting at each token in the document.
    Combined, this takes O(MK + NK) time and up to O(MK) space.
    """

    class TrieNode:
        """
        TrieNode that stores a token, matches that terminate at the token,
        and links to further tokens in the Trie.
        """
        def __init__(self, token: str):
            self.token = token
            self.trieMatches = set()
            self.trieNodes = {}
        def getToken(self):
            # Retrieve token represented by this TrieNode.
            return self.token
        def getTrieLeaves(self) -> set[str]:
            # Retrieve all matches that terminate at self.token.
            return self.trieMatches
        def addTrieLeaf(self, matchEntity: str):
            # Add match string terminates at self.token.
            self.trieMatches.add(matchEntity)
        def getTrieNode(self, token: str):
            # For the token, return the child TrieNode.
            # If not found, return None.
            return self.trieNodes.get(token, None)
        def addTrieNode(self, trieNode):
            # Map token to child TrieNode for searching.
            newToken = trieNode.getToken()
            if newToken not in self.trieNodes:
                self.trieNodes[newToken] = trieNode

    class TrieEntity:
        """
        Wrapper class for tokenized and matched entities
        via tokenization or searching the Trie.
        """
        def __init__(self, entity: str, startIdx: int, endIdx: int):
            self.entity = entity
            self.start = startIdx
            self.end = endIdx
        def __hash__(self):
            # Compute hash for TrieMatch.
            p = 31
            hashData: list[Hashable] = [self.entity, self.start, self.end]
            return sum([x.__hash__() * pow(p, i) for i, x in enumerate(hashData)])
        def __repr__(self):
            return f"{{Entity: {self.entity}, Start Index: {self.start}, End Index: {self.end}}}"
        def getEntity(self) -> str:
            return self.entity
        def getStartIdx(self) -> int:
            return self.start
        def getEndIdx(self) -> int:
            return self.end

    class Tokenizer(ABC):
        """
        Abstract base class for Tokenizers.
        """
        @abstractmethod
        def tokenize(self, doc: str):
            """
            Tokenize a string into a list of TrieEntity.
            """

    class CharTokenizer(Tokenizer):
        """
        Tokenizer that splits documents into characters.
        Default Tokenizer if no other Tokenizer is specified.
        """
        @staticmethod
        def tokenize(doc: str):
            """
            Tokenize document into non-empty characters.
            """
            return [TrieTagger.TrieEntity(c, i, i+1) for i, c in enumerate(doc)]

    class WordTokenizer(Tokenizer):
        """
        Tokenizer that splits documents into words, i.e.
        sequences of characters bounded by whitespace.
        """
        SPACE = " "
        @classmethod
        def tokenize(cls, doc: str):
            """
            Tokenize document into non-empty word strings.
            """
            tokenList = []
            spaceSplit = doc.split(cls.SPACE)
            posIdx = 0
            for w in spaceSplit:
                if w:
                    # Append new TrieEntity.
                    tokenList.append(TrieTagger.TrieEntity(w, posIdx, posIdx + len(w)))
                    # Increment posIdx by len(w) + 1.
                    posIdx += len(w) + 1
                else:   # Skip empty strings.
                    # Increment posIdx by 1 for a single whitespace.
                    posIdx += 1
            # Return tokenized document.
            return tokenList

    def __init__(self, entityList: list[str], tokenizer: Tokenizer = None):
        """
        Instantiate the TrieTagger.
        """
        # Setup tokenizer.
        self.tokenizer: TrieTagger.Tokenizer = tokenizer
        # Root of the Trie.
        self.root: TrieTagger.TrieNode = TrieTagger.TrieNode("")
        # Insert all matchStrings into the Trie.
        self.insertEntity(entityList)

    def tokenize(self, doc: str) -> list[str]:
        """
        Executes the configured tokenizer for the TrieTagger.
        """
        if self.tokenizer is not None:
            # Utilize specified Tokenizer.
            return self.tokenizer.tokenize(doc)
        else:
            # Default to Character Tokenizer.
            return TrieTagger.CharTokenizer.tokenize(doc)
        
    def tag(self, document: str):
        """
        Tag and identify all entities detected in the document
        that have been registered into the TrieTagger.
        """
        # Tokenize the document.
        docTokens: list[TrieTagger.TrieEntity] = self.tokenize(document)
        # Iterate through all starting tokens of the document.
        trieMatches = []
        for i in range(len(docTokens)):
            # Search for TrieNodes in the Trie. Track position in document.
            tokenIdx = 0
            startIdx = docTokens[i].getStartIdx()
            curNode = self.root
            while i + tokenIdx < len(docTokens):
                # Retrieve the token and TrieNode.
                tokenEntity: TrieTagger.TrieEntity = docTokens[i+tokenIdx]
                tokenNode: TrieTagger.TrieNode = curNode.getTrieNode(tokenEntity.getEntity())
                # Terminate search if no more matched tokens.
                if tokenNode is None:
                    break
                # Append terminal TrieMatch to the output list.
                trieMatches.extend([
                    TrieTagger.TrieEntity(e, startIdx, tokenEntity.getEndIdx())
                    for e in tokenNode.getTrieLeaves()
                ])
                # Continue traversing the Trie.
                curNode = tokenNode
                # Increment token index.
                tokenIdx += 1
        # Return all matched entities in the Trie.
        return trieMatches

    def insertEntity(self, entityList: list[str]):
        """
        Insert all strings in entityList into the Trie.
        """
        # Insert all entities into the Trie.
        for entity in entityList:
            # Tokenize.
            tokenEntityList = self.tokenize(entity)
            # Sift through the Trie.
            tokenIdx = 0
            curNode = self.root
            while tokenIdx < len(tokenEntityList):
                # Retrieve the token and TrieNode.
                tokenEntity: TrieTagger.TrieEntity = tokenEntityList[tokenIdx]
                tokenNode = curNode.getTrieNode(tokenEntity.getEntity())
                # Existing token.
                if tokenNode is not None:
                    # Continue traversing the Trie.
                    curNode = tokenNode
                else:   # New token!
                    # Create new TrieNode.
                    newNode = TrieTagger.TrieNode(tokenEntity.getEntity())
                    # Map current TrieNode to new TrieNode.
                    curNode.addTrieNode(newNode)
                    # Continue traversing the Trie.
                    curNode = newNode
                # Move to next token to insert.
                tokenIdx += 1
            # Reached the end of the entity, add the entity
            # as a TrieMatch in the terminal token's TrieNode.
            curNode.addTrieLeaf(entity)


class HeapCache:
    """
    Implementation of an ordered cache / HeapCache, such that
    the cache pops the min-order or max-order data when the cache
    no longer has the capacity. When reverse=False, this implements
    a min heap, and when reverse=True, this implements a max heap.
    For example, HeapCache(reverse=False) where the order represents
    timestamps implements a least-recently used cache (LRUCache).

    Supports "static" use cases of internal methods if an external
    cache and KV are provided.
    """

    class Data:
        """
        Generic Dataclass for HeapCache.
        """
        def __init__(self, key: Hashable, data, order, index=None):
            self.key = key
            self.data = data
            self.order = order
            self.index = index
        def getKey(self):
            return self.key
        def getData(self):
            return self.data
        def setData(self, data):
            self.data = data
        def getOrder(self):
            return self.order
        def setOrder(self, order):
            self.order = order
        def getIndex(self):
            return self.index
        def setIndex(self, index):
            self.index = index

    def __init__(self, reverse=False, capacity=None):
        """
        Instantiate the KV store and Heap.
        """
        self.KV = {}
        self.heapCache = []
        self.reverse = reverse
        self.capacity = capacity

    def __repr__(self, rawHeap=True):
        """
        String representation of the heap.
        """
        if rawHeap:
            # Print raw heap without sorting.
            return str([f"({x.getKey()}, {x.getData()}, {x.getOrder()}, {x.getIndex()})" for x in self.heapCache])
        else:
            # HeapSort
            return str([f"({data.getData()}, {data.getOrder()})" for data in self.heapsort()])

    def heapsort(self, kv: dict = None):
        """
        Execute HeapSort.
        """
        # Custom KV associated with a HeapCache.
        keyValue = self.KV if kv is None else kv
        # Execute HeapSort.
        cacheCopy = []
        kvCopy = {}
        # Clone the HeapCache.
        for key, value in sorted(keyValue.items(), key=lambda x: x[1].getIndex(), reverse=False):
            kvCopy[key] = self.Data(key, value.getData(), value.getOrder(), value.getIndex())
            cacheCopy.append(kvCopy[key])
        # Sort.
        output = []
        while cacheCopy:
            output.append(self.popData(cache=cacheCopy, kv=kvCopy))
        return output

    def insertData(self, key: Hashable, value, order: int, cache: list = None, kv: dict = None):
        """
        Insert data into HeapCache. If an existing data with the
        same key already exists, overwrite the data from the heap.

        Furthermore, if the capacity of the HeapCache is specified
        and is exceeded, pop an element from HeapCache.
        """
        # Custom cache and KV.
        heapCache = self.heapCache if cache is None else cache
        keyValue = self.KV if kv is None else kv
        # Check if key exists in KV / Heap.
        if key in keyValue:
            # Overwrite.
            keyValue[key].setData(value)
            keyValue[key].setOrder(order)
            # Re-heapify.
            self.sift(keyValue[key].getIndex(), cache=heapCache)
        else:
            # Create new Data.
            data = self.Data(key, value, order, index=len(heapCache))
            # Insert into KV.
            keyValue[key] = data
            # Append to HeapCache.
            heapCache.append(data)
            # Sift up.
            self.sift(data.getIndex(), cache=heapCache)
            # Capacity checking.
            if isinstance(self.capacity, int) and len(heapCache) > self.capacity:
                overflow = len(heapCache) - self.capacity
                for _ in range(overflow):
                    # Pop element from heap.
                    self.popData(cache=heapCache, kv=keyValue)

    def popData(self, key: Hashable = None, cache: list = None, kv: dict = None):
        """
        Pop specified element and heapify. If no key is provided,
        then pop the initial element of the heap.
        """
        # Custom cache and KV.
        heapCache = self.heapCache if cache is None else cache
        keyValue = self.KV if kv is None else kv
        if not heapCache:
            # Empty HeapCache. Return None.
            return None
        
        # Pop specific key from HeapCache and KV.
        targetIdx = 0
        if key in keyValue:
            # Search heap index by key.
            targetIdx = keyValue[key].getIndex()
            # Pop key from KV.
            keyValue.pop(key)
        elif key is None:
            # Search key by heap index.
            key = heapCache[targetIdx].getKey()
            # Pop key from KV.
            keyValue.pop(key)
        else:
            # Key is specified but does not exist. Do nothing.
            return None

        # Identify element at the specified index of the heap.
        data = heapCache[targetIdx]
        # Heapify (if there exists at least one child).
        if len(heapCache)-1 > targetIdx:
            # Swapping with the bottom of
            # the heap preserves indices.
            heapCache[targetIdx] = heapCache.pop()
            heapCache[targetIdx].setIndex(targetIdx)
            # Sift down if children exist.
            self.sift(targetIdx, cache=heapCache)
        else:
            # Empty the first and only element from the heap.
            heapCache.pop(targetIdx)
        # Return data.
        return data
    
    def heapify(self, cache: list = None):
        """
        Heapify the cache.
        """
        # Custom cache and KV.
        heapCache = self.heapCache if cache is None else cache
        # Heapify via iterative sift down.
        for i in range(len(heapCache)-1, -1, -1):
            # Sift down from the end of the heap.
            self.sift(i, cache=heapCache)

    def sift(self, siftIdx: int, cache: list = None):
        """
        Initiate a bi-directional sift operation optionally starting at siftIdx.
        If the parent violates the heap, sift up. If a child violates the heap, sift down.
        Update swapped indices along the way.
        """
        # Custom cache and KV.
        heapCache = self.heapCache if cache is None else cache
        # Sift up or down to heapify.
        while siftIdx >= 0 and siftIdx < len(heapCache):
            # Compute least- or most-recently used data in binary heap.
            swapIdx = siftIdx
            for adjIdx in [2*siftIdx+1, 2*siftIdx+2, math.floor((siftIdx - 1) / 2)]:
                if adjIdx < 0 or adjIdx >= len(heapCache):
                    # No children or no parent.
                    continue
                elif any([
                    not self.reverse and any([
                        # Parent has higher order than child, which requires sifting.
                        adjIdx < swapIdx and heapCache[adjIdx].getOrder() > heapCache[swapIdx].getOrder(),
                        # Child has lower order than parent, which requires sifting.
                        adjIdx > swapIdx and heapCache[adjIdx].getOrder() < heapCache[swapIdx].getOrder()
                    ]),
                    self.reverse and any([
                        # Parent has lower order than child, which requires sifting.
                        adjIdx < swapIdx and heapCache[adjIdx].getOrder() < heapCache[swapIdx].getOrder(),
                        # Child has higher order than parent, which requires sifting.
                        adjIdx > swapIdx and heapCache[adjIdx].getOrder() > heapCache[swapIdx].getOrder()
                    ])
                ]):
                    # Identify higher or lower order data.
                    swapIdx = adjIdx
            if swapIdx != siftIdx:
                # Sift.
                heapCache[swapIdx], heapCache[siftIdx] = heapCache[siftIdx], heapCache[swapIdx]
                # Update indices for insertion and deletion at O(log(n)).
                heapCache[swapIdx].setIndex(swapIdx)
                heapCache[siftIdx].setIndex(siftIdx)
                # Iterate in the direction of the sift. Will converge or break.
                siftIdx = swapIdx
            else:
                # No swap necessary, so heap property is satisfied.
                break


class QuickSort:
    """
    QuickSort implementation with Lomuto "median-of-three" partitioning.
    """

    class Comparable(metaclass=ABCMeta):
        """
        ABC for enforcing __lt__ comparability.
        """
        @abstractmethod
        def __lt__(self, other: Any) -> bool: ...

    CT = TypeVar("CT", bound=Comparable)

    @staticmethod
    def sort(seq: MutableSequence[CT], reverse: bool = False):
        """
        Static function for executing in-place QuickSort on the MutableSequence provided.
        """
        if seq:
            # Instantiate QuickSort boundaries.
            low = 0
            high = len(seq) - 1
            # QuickSort
            QuickSort.quicksort(seq, low, high)
        if reverse:
            seq.reverse()

    @staticmethod
    def quicksort(seq: MutableSequence[CT], low: int, high: int):
        """
        Recursive Lomuto "median-of-three" QuickSort implementation.
        :param seq <MutableSequence<CT>>:   Sequence of ComparableType that have defined ordering.
        :param low <int>:                   Index of lower bound of sorting scope.
        :param high <int>:                  Index of upper bound of sorting scope.
        """
        # Continuation criteria.
        if low >= 0 and high >= 0 and low < high:
            # Partition.
            lt, gt = QuickSort.partition(seq, low, high)
            # Sort lower partition.
            QuickSort.quicksort(seq, low, lt-1)
            # Sort upper partition.
            QuickSort.quicksort(seq, gt+1, high)

    @staticmethod
    def partition(seq: MutableSequence[CT], low: int, high: int):
        """
        Partition sequence into elements less or greater than a median-of-three pivot.
        :param seq <MutableSequence<CT>>:   Sequence of ComparableType that have defined ordering.
        :param low <int>:                   Index of lower bound of sorting scope.
        :param high <int>:                  Index of upper bound of sorting scope.
        """
        # Compute median pivot and swap to mid.
        mid = math.floor(low + (high - low) / 2)    # To avoid integer overflow from adding low and high.
        QuickSort.centerMedian(seq, low, mid, high)
        median = seq[mid]

        # Swap the elements around the pivot.
        lt = low    # Lowest upper bound of the lesser partition.
        eq = low    # Lowest upper bound of the equivalent partition. Always lt <= eq.
        gt = high   # Greatest lower bound of the greater partition.
        while eq <= gt:
            if seq[eq] < median:
                # Swap to lesser partition.
                QuickSort.swap(seq, eq, lt)
                # Extend the lesser partition boundary.
                lt += 1
                # Extend the equiv partition boundary to
                # account for the new addition into the
                # lesser partition which increments the
                # position of the equiv partition.
                eq += 1
            elif seq[eq] > median:
                # Swap to greater partition.
                QuickSort.swap(seq, eq, gt)
                # Extend the greater partition boundary.
                gt -= 1
            else:   # seq[eq] == median
                # Extend the equiv partition boundary.
                eq += 1
        # Return lowest upper bound and greatest lower bound of the unsorted sequence.
        return lt, gt
    
    @staticmethod
    def centerMedian(seq: MutableSequence[CT], low: int, mid: int, high: int):
        """
        Compute the median-of-three for low, mid and high in seq.
        After sorting, the median is swapped into the mid spot.
        :param seq <MutableSequence<CT>>:   Sequence of ComparableType that have defined ordering.
        :param low <int>:                   Lowest element.
        :param mid <int>:                   Median element.
        :param high <int>:                  Highest element.
        """
        # Sort low, mid, and high in-place.
        if seq[low] > seq[mid]:
            # Swap low and mid.
            QuickSort.swap(seq, low, mid)
        if seq[mid] > seq[high]:
            # Swap mid and high.
            QuickSort.swap(seq, mid, high)
        if seq[low] > seq[mid]:
            # Swap low and mid (again).
            QuickSort.swap(seq, low, mid)

    @staticmethod
    def swap(seq: MutableSequence[CT], left: int, right: int):
        """
        Swap the elements at index left and right in-place.
        :param seq <MutableSequence<CT>>:   Sequence of ComparableType that have defined ordering.
        :param left <int>:                  Index of left element.
        :param right <int>:                 Index of right element.
        """
        # Swap left and right.
        seq[right], seq[left] = seq[left], seq[right]

class LinkedList:
    """
    Linked list implementation with common transformations.
    No practical use besides brain-teasing.
    """

    class Node:
        """
        LinkedList Node
        """
        def __init__(self, data):
            self.data = data
            self.next = None
        def __repr__(self):
            return f"{self.data}"

    def __init__(self):
        """
        Instantiate (non-empty) LinkedList.
        """
        self.head = self.Node("HEAD")

    def __repr__(self):
        """
        Print LinkedList.
        """
        listOutput = []
        nodeIter = self.head
        while nodeIter is not None:
            listOutput.append(nodeIter)
            nodeIter = nodeIter.next
        return " -> ".join([f"{x.data}" for x in listOutput])
    
    def iterate(self, terminateCondition: Callable[..., bool]):
        """
        Iterate until termination condition is satisfied.
        """
        nodeIter = self.head
        while nodeIter is not None and not terminateCondition(nodeIter):
            nodeIter = nodeIter.next
        return nodeIter

    def append(self, node: Node):
        """
        Append Node to LinkedList.
        """
        # Search for the final node in the LinkedList.
        finalNode = self.iterate(lambda x: x.next is None)
        finalNode.next = node
        return node
    
    def delete(self, data):
        """
        Delete the initial node containing data.
        Return data if deleted, else return None.
        """
        # Search for node before the node with matching data.
        prevSearchNode = self.iterate(lambda x: x.next is not None and x.next.data == data)
        # Delete the node if not None.
        if prevSearchNode is not None:
            prevSearchNode.next = prevSearchNode.next.next
            return data
        else:
            # Iterated to the end of the LinkedList. Do not delete.
            return None
        
    def clear(self):
        """
        Clear the LinkedList.
        """
        # Create a new HEAD node.
        self.head = self.Node("HEAD")

    def swap(self, data1, data2):
        """
        Swap two nodes in the LinkedList.
        
        For example, swap(2,4) ~ swap(4,2) implies:

        {1 -> [2] -> 3 -> [4] -> 5}  =>  {1 -> [4] -> 3 -> [2] -> 5}

                     |                                ^
                     v                                |

        {1 -> [2]    3 <> [4]    5}  =>  {1    [2] <- 3 <- [4]    5}
               |-----------------^        |     |-----------------^
                                          |-----------------^
        """
        # Search for the nodes before the two nodes to swap.
        prevFirstNode = self.iterate(lambda x: x.next is not None and x.next.data == data1)
        prevSecondNode = self.iterate(lambda x: x.next is not None and x.next.data == data2)
        if prevFirstNode is None or prevSecondNode is None:
            # At least one of the nodes does not exist. Do nothing.
            raise LookupError("At least one of the nodes specified does not exist and cannot be swapped.")
        
        # Swap next node pointers.
        tempFirstNext = prevFirstNode.next.next
        prevFirstNode.next.next = prevSecondNode.next.next
        prevSecondNode.next.next = tempFirstNext

        # Swap prev node pointers.
        tempFirst = prevFirstNode.next
        prevFirstNode.next = prevSecondNode.next
        prevSecondNode.next = tempFirst

    def reverse(self):
        """
        Reverse the LinkedList.
        """
        # Iterate through the list, reversing each of the next pointers.
        nodeIter = self.head.next
        prevIter = None
        nextIter = None
        while nodeIter is not None:
            # Save next node to iterate to.
            nextIter = nodeIter.next
            # Reverse direction of list.
            nodeIter.next = prevIter
            # Track current node as next previous node for reversal.
            prevIter = nodeIter
            # Iterate to next node.
            nodeIter = nextIter
        # Reset the HEAD node to point to the final previous node.
        self.head.next = prevIter


class FibonacciCache:
    """
    Compute all Fibonacci numbers. Optionally, cache them for future calculations.
    """

    def __init__(self):
        """
        Instantiate the FibonacciCache.
        """
        self.fiboCache = {0: 0, 1: 1}

    def __repr__(self):
        """
        Print the largest Fibonacci number stored in this instance.
        """
        return f"Fibonacci Cache Element Number {len(self.fiboCache)} : {self.fiboCache[len(self.fiboCache) - 1]}"

    def fibonacci(self, n: int, cache: bool = True):
        """
        Compute all Fibonacci numbers up to n.
        :param n <int>:         Compute the n-th Fibonacci number.
        :param cache <bool>:    Used cached implementation.
        """
        if cache:
            return self.cachedFibonacci(n)
        else:
            return self.recursiveFibonacci(n)
    
    def cachedFibonacci(self, n: int):
        """
        Compute Fibonacci numbers using a cache.
        :param n <int>:     Compute and cache n Fibonacci numbers.
        """
        for i in range(n):
            if i >= len(self.fiboCache):
                # Inductively compute Fibonacci numbers.
                self.fiboCache[i] = self.fiboCache[i-1] + self.fiboCache[i-2]
        # Return requested Fibonacci number.
        return self.fiboCache[len(self.fiboCache) - 1]
    
    def recursiveFibonacci(self, n: int):
        """
        Compute Fibonacci numbers via recursion.
        :param n <int>:     Compute the n-th Fibonacci number.
        """
        if n == 0 or n == 1:
            return n
        else:
            # Recursively compute Fibonacci numbers.
            self.fiboCache[n] = self.recursiveFibonacci(n-1) + self.recursiveFibonacci(n-2)
            return self.fiboCache[n]

class TarjanSCC:
    """
    Compute all strongly-connected components in a directed graph G.
    Utilizes Tarjan's strongly-connected components recursion DFS algorithm.
    Returns a list of strongly-connected components sorted topologically.
    """

    def __init__(self, graph):
        """
        Instantiate graph information for strongly-connected component searching of G.
        :param graph <list<list>>:  Adjacency matrix for the graph G. Nodes are indexed by 
                                    non-negative integers, i.e. 0, 1, 2, ...
        """

        self.G = graph                  # Adjacency Matrix for Graph
        self.dfs = []                   # Depth-First Search Stack
        self.index = 0                  # Exploration Index
        self.D = {                      # Node Data
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
        self.dfs.append(v)
        self.D[v]['instack'] = True
        
        # Explore adjacent nodes.
        for w in range(len(self.G[v])):
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
                # Explored node in the DFS stack. (Back-Edge Node)
                elif self.D[w]['instack']:
                    # Update the minimum exploration index relative to
                    # the back-edge node index. Do NOT utilize the minimum 
                    # reachable exploration index of the back-edge node, 
                    # which considers minimum reachable exploration indices 
                    # of the sub-trees of the back-edge node!
                    self.D[v]['minlink'] = min(
                        self.D[v]['minlink'],
                        self.D[w]['index']
                    )
        
        # Output the SCC if the node is a minimal reachable node of the SCC.
        scc_detect = []
        if self.D[v]['minlink'] == self.D[v]['index']:
            # Include nodes in the sub-tree of the minimal reachable node.
            while self.dfs and self.D[self.dfs[-1]]['index'] >= self.D[v]['index']:
                w = self.dfs.pop()
                scc_detect.append(w)
                self.D[w]['instack'] = False

        return scc_detect


class DijkstraBFS:

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
            # Only search from the initial node instead. More efficient.
            task = [initial_node]
        for v in task:
            # Reset queue and processed set.
            heap = []
            # FIFO Queue for BFS. Using a min heap 
            # to sort by edge weight.
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
    

class DisjointEnsemble:
    """
    Implementation of a DisjointEnsemble data structure that
    specifically supports efficient union of disjoint sets
    stored within the ensemble of sets.
    """
    class TreeNode:
        def __init__(self, data: Hashable, parent: int = None, children: list = None, rank: int = 0):
            self.data = data
            self.parent = parent
            self.children = [] if children is None else children
            self.rank = rank
        def __eq__(self, x):
            # Equate on element ID.
            return self.data == x.data
        
    def __init__(self, space: Iterable[Hashable]):
        # Initialize TreeNode for all elements in the space.
        self.treeCache = {
            x: DisjointEnsemble.TreeNode(x) for x in space
        }
        
    def findTreeRoot(self, node: Hashable):
        """
        Search for the root of the spanning tree containing node.
        """
        # Lookup the TreeNode.
        treeNode = self.treeCache.get(node, None)
        if treeNode is None:
            # Empty tree. Return None.
            return DisjointEnsemble.TreeNode(None)
        
        # Search for the root.
        root = treeNode
        while root.parent is not None:
            # Iterate to the parent.
            root = root.parent

        # Update parent pointers to root for lookup to O(1).
        nodeIter = treeNode
        while nodeIter.parent is not None:
            # Flatten the search tree.
            tempParent = nodeIter.parent
            nodeIter.parent = root
            root.children.append(nodeIter)
            nodeIter = tempParent
        
        # Return the root node for spanning tree comparison.
        return root

    def treeUnion(self, a: Hashable, b: Hashable):
        """
        Unify the spanning trees containing nodes A and B.
        """
        # Search for the roots of A and B.
        rootA = self.findTreeRoot(a)
        rootB = self.findTreeRoot(b)
        if rootA == rootB:
            # Same spanning tree. No union required.
            return

        # Compare or update the rank of the roots.
        if rootB.rank > rootA.rank:
            # Swap to set new root as B.
            rootA, rootB = rootB, rootA
        elif rootA.rank == rootB.rank:
            # Update rank to rootB.rank + 1.
            rootA.rank += 1

        # Unify the spanning trees.
        rootB.parent = rootA
        rootA.children.append(rootB)
        return
    
    def getTree(self, node: Hashable):
        """
        Retrieve the disjoint set associated with node.
        """
        # Find the root.
        root = self.findTreeRoot(node)
        if root is None:
            return set()
        
        # Depth-First Tree Search
        stack = [root]
        output = set()
        while stack:
            # Pop the stack.
            node = stack.pop()
            # Add node content to output set.
            output.add(node.data)
            # Append children to stack.
            stack.extend(node.children)
        return output

class KruskalMST:
    """
    Implementation of Kruskal's Minimal Spanning Tree Algorithm.
    Utilizes DisjointEnsemble to iteratively unify multiple
    trees to construct a minimal (or maximal) spanning tree.
    """

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
        # Insert weighted edge into priority heap / queue.
        for i in range(len(graph)):
            for j in range(len(graph)):
                if graph[i][j] != 0:    # Non-existent edge.
                    heapq.heappush(
                        self.E,
                        (graph[i][j], (i,j)) if not maximal else (-graph[i][j], (i,j))
                    )
        # Register nodes as TreeNode for root-searching.
        self.disjointTrees = DisjointEnsemble(range(len(self.G)))
        # Control minimal or maximal spanning tree algorithm.
        self.maximal = maximal

    def mst(self):
        """
        Compute a list of edges that constitutes the minimal spanning tree of the graph G.
        Return list of edges constituting minimal spanning tree, and the cumulative tree edge weight score.
        """

        # Build minimal spanning tree.
        tree = []
        score = 0
        while self.E:

            # Pop the minimal edge.
            w, e = heapq.heappop(self.E)

            # Combine sets of edges if the trees associated
            # to each vertex of edge e are not equivalent,
            # preventing cycles from being created.
            if self.disjointTrees.findTreeRoot(e[0]) != self.disjointTrees.findTreeRoot(e[1]):

                # Union the trees to create a larger spanning tree.
                self.disjointTrees.treeUnion(e[0], e[1])

                # Append edge to MST.
                tree.append(e)
                if not self.maximal:
                    score += w
                else:
                    score -= w

        return tree, score

class KnapSack:

    def __init__(self, value: list[float], cost: list[int], weight=None, repetition=False):
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
        self.Q = {  # Initialize reward matrix of shape (weight, items).
            # Item -1 represents item repetition for each weight,
            # tracking the highest value knapsack by relaxing
            # the constraint that knapsacks must build from
            # previous chosen items. Instead, it just builds
            # off the previous weight's maximum value knapsack.
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
                    # Cannot even add item to an empty knapsack
                    # without overflowing the weight limit.
                    # Set to knapsack not including item k, i.e.
                    # persisting the same weight w.
                    self.Q[w][k if not self.rep else -1] = self.Q[w][k-1 if not self.rep else -1]
                else:
                    # Analyze reward from adding new item to knapsack.
                    test_val = self.Q[w-self.cost[k]][k-1 if not self.rep else -1][0] + self.value[k]
                    # If the reward is greater than the highest value knapsack of the same weight...
                    if test_val > self.Q[w][k-1 if not self.rep else -1][0]:
                        # Include new item. Update knapsack.
                        self.Q[w][k if not self.rep else -1] = (
                            test_val,
                            self.Q[w-self.cost[k]][k-1 if not self.rep else -1][1] + [k]
                        )
                    else:
                        # Exclude new item. Continue using the knapsack with the same weight.
                        self.Q[w][k if not self.rep else -1] = self.Q[w][k-1 if not self.rep else -1]
                # Update optimal knapsack.
                if self.Q[w][k if not self.rep else -1][0] > Q_opt[0]:
                    Q_opt = self.Q[w][k if not self.rep else -1]

        return Q_opt


class LevenshteinDP:

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


class WaterCapture:
    """
    [Greedy Water Capture Problem]
    Given a collection of wall(s) / barrier(s) with various height, 
    compute the optimal rectangular container represented by precisely 
    2 wall(s) / barrier(s) that capture the most rainwater measured by 
    the cross-sectional area of the container.

    |~~~~~~~~~~~~~~         
    | Water Area=9 |
    |    |         |
    |____|____|____|____.____|

    Alternatively, we can consider an alternative model where the walls
    have volume that can take up space and the objective is to compute
    the total volume of water captured by this structure:

    X
    X ~ ~ X
    X X ~ X
    X X X X ~ X

    which can hold 4 units of water across the entire structure.
    """

    def __init__(self, height_vector):
        """
        Initialize parameters to solve the greedy water capture problem 
        given a vector of wall heights.
        :param height_vector <list<float>>: List of wall heights.
        """
        
        # Store container information.
        self.bars = height_vector

    def water_volume(self):
        """
        Compute the optimal volume of water captured by a container formed from the wall(s) / barrier(s).

        Intuitively, the minimum bar height of a container dictates the cross-sectional area, so searching
        for a different bar on the side of the container with greater height only decreases the cross-sectional
        area via decreasing the width of the tested container. By searching for a higher bar on the side with
        lower height and testing for optimality, we can deduce the maximal container.
        """
        # Instantiate search pointers.
        l = 0
        r = len(self.bars)-1
        x_area = 0

        # Loop over all width(s) of all possible containers
        # from largest (len(self.bars) - 1) to smallest (1).
        for width in range(len(self.bars)-1, 0, -1):

            # Greedy search for maximum volumne.
            if self.bars[l] < self.bars[r]:
                # Track cross-sectional area.
                x_area = max(x_area, self.bars[l] * width)
                # Test different container.
                l += 1
            else:
                # Track cross-sectional area.
                x_area = max(x_area, self.bars[r] * width)
                # Test different container.
                r -= 1

        return x_area
    
    def water_volume_alt(self):
        """
        Compute the total amount of water caught by landscape of blocks.
        
        Consider that the amount of water trapped at any vertical slice
        of the structure can be computed as:

        { min(Max Left Height, Max Right Height) - Current Height }

        such that we can integrate from the left and right via dynamically
        updating the minimum boundary height to evaluate the height of
        the water that can be trapped within a slice.
        """

        # Iterators. Start internally as impossible to
        # store water on edge of structure.
        l = 1
        r = len(self.bars)-2
        
        # Track maximum left and right height.
        maxLeft = self.bars[0]
        maxRight = self.bars[len(self.bars)-1]

        # Integrate while width is non-zero.
        volume = 0
        while l <= r:
            if maxLeft < maxRight:
                # Compute water slice volume.
                v = maxLeft - self.bars[l]
                if v > 0:
                    # Non-negative volume of trapped water.
                    volume += v
                # Update maxLeft.
                if self.bars[l] > maxLeft:
                    maxLeft = self.bars[l]
                # Increment l.
                l += 1
            else:   # maxLeft >= maxRight
                # Compute water slice volume.
                v = maxRight - self.bars[r]
                if v > 0:
                    # Non-negative volume of trapped water.
                    volume += v
                # Update maxRight.
                if self.bars[r] > maxRight:
                    maxRight = self.bars[r]
                # Increment r.
                r -= 1
        
        # Return integrated volume.
        return volume
                

class Numerics:
    
    @staticmethod
    def randNFromRandK(n: int, k: int):
        """
        Generate a uniformly random integer from [n]
        given a random sampling granularity of [k].
        """
        # Execute randInt(k) multiple times and assign combinations to [0,6].
        while True:
            # Sample k-nary coefficients from 0 to k-1.
            subSample = []
            for _ in range(n // k + 1):
                subSample.append(random.randint(0,k-1))
            # Decode the k-nary number from the k-nary representation.
            p_k = sum([pow(k, i) * subSample[i] for i in range(len(subSample))])
            # Delete non-defined combinations, i.e. numbers greater than
            # the largest multiple of n less than the largest possible sample x.
            x = sum([pow(k, i) * (k-1) for i in range(len(subSample))])
            if p_k < x - x % n:
                # Return randN.
                return p_k % n
    
    @staticmethod
    def randomSample(n: int, k: int):
        """
        Uniformly randomly sample exactly k elements from [n] = {0, ..., n-1}.
        When k = n, the algorithm randomly shuffles [n].

        Algorithm iteratively builds on swaps to create permutations, such that
        newly processed elements (i.e. at index j) swap positions with pre-processed
        elements (i.e. with index in [0, j-1]) with probability 1/j. Integrating
        these choices produces a uniform permutation probability of 1/n! when k = n,
        which is a random ordering or 'shuffle' of [n].

        When k < n, each k-permutation created by truncating the n-permutation
        to k initial elements exist with uniform probability (n-k)! / n!, i.e. for
        each k-permutation there exist (n-k)! permutations of the final (n-k) elements,
        so the probability of each k-permutation is muliplied by (n-k)!. Furthermore,
        we can get all k-combinations by ignoring the order of the initial k elements,
        i.e. for each k-combination there exist k! permutations of the k elements, so
        the probability of a k-combination is multiplied by k!. Thus, the probability
        of selecting any k-combination is precisely and uniformly k! (n-k)! / n!, the
        reciprocal of the combinatorial (n,k). Thus, a uniform random sample of [n].
        """
        # Instantiate the sample of k elements taken from the
        # initial k elements of the complete population.
        population = [i for i in range(0, n)]
        # Iterate through the array, randomly sampling a number
        # in [0, j-1] such that the element at that sampled index
        # is replaced / swapped with the element at j.
        for j in range(0, n):
            idx = random.randint(0, j)
            if idx < j:
                # Swap. 
                population[idx], population[j] = population[j], population[idx]
        return population[0:k]
    
    @staticmethod
    def triangleAverage(l: list[float]):
        """
        Compute all moving averages of l.
        """
        avgList = []
        m = 0
        n = 0
        for x in l:
            # Compute moving average.
            m = Numerics.movingAverage(x, m, n)
            n += 1
            avgList.append(m)
        return avgList
    
    @staticmethod
    def triangleVariance(l: list[float]):
        """
        Compute all moving variances of l.
        """
        varList = []
        m = 0
        v = 0
        n = 0
        for x in l:
            # Compute moving variance.
            v = Numerics.movingVariance(x, m, v, n)
            # Compute moving average.
            m = Numerics.movingAverage(x, m, n)
            n += 1
            varList.append(v)
        return varList

    @staticmethod
    def movingAverage(x: float, m: float, n: int):
        """
        Provided a new element x and a moving average m of n elements,
        compute the new average including x.
        """
        # Compute the moving average.
        return (m * n + x) / (n + 1)
    
    @staticmethod
    def movingVariance(x: float, m: float, v: float, n: int):
        """
        Provided a new element x, a moving average m, and moving
        variance v of n elements, compute the new variance including x.
        """
        return v * n / (n+1) + pow(x-m, 2) * n / pow(n+1, 2)