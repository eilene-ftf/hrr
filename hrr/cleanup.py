import numpy as np
from .hrr import HRR, HRRArray
from typing import Callable
from numpy.typing import ArrayLike, DTypeLike

class CleanupMemory:
    """Stub class for generic cleanup memories.

    Attributes:
        n (int): Number of stored vectors (may change); may differ from reserved memory.
        d (int): Number of dimensions of stored vectors (may not be changed).
        dtype (dtype): Type of the vectors stored in cleanup memory.
        sim (Callable): Function that computes the similarity between two vectors in case they 
                        aren't HRRs. Default cosine similarity.
    """

    def __init__(self, n:int, d:int, *args, 
                 dtype:DTypeLike=np.float64, 
                 sim:Callable[[ArrayLike, ArrayLike], DTypeLike] = lambda u, v: (u@v)/(u@u * v@v),
                 **kwargs
                 ):
        """Initialize an n by d cleanup memory.

        Args:
            n (int): Number of vectors in the cleanup memory.
            d (int): Dimension of vectors to be stored in the memory (must be constant).
            dtype (DTypeLike): numpy dtype for the memory.
        
        Returns:
            None
        """

        self.dtype = np.dtype(dtype) # unused for now, may be necessary with FHRRs
        self.n = n
        self.d = d
        self.sim = sim

    def nearest(self, 
                probe:np.ndarray | HRR, 
                k:int=1
                ) -> tuple[HRRArray, np.ndarray]:
        """Returns k nearest neighbours of probe using probe's similarity measure or self.sim 
        function if unavailable (default 1), and similarities of neighbours, in descending order.

        Args:
            probe (np.ndarray | HRR): The vector whose neighbours you want to find.
            k (int): Number of neighbours to find.

        Returns:
            HRRArray: The top k HRRs in the search (greatest similarity first).
            ndarray: The similarities of the top k items
        """

        return HRRArray(np.zeros((k, self.d))), np.zeros(k)

    def insert(self,
               trace: np.ndarray | HRR
               ):
        """Adds a new HRR into the memory, expanding it if necessary.
        
        Args:
            trace (np.ndarray | HRR): the vector to add to memory.

        Returns:
            None
        """
        pass

    def __getitem__(self, probe:HRR, showsim:bool=False
                    ) -> HRR | tuple[HRR, DTypeLike]:
        """Fetches the nearest neighbour of the probe as a dictionary lookup.

        Args:
            probe (HRR): the HRR whose neighbour we're getting.
            showsim (bool): whether to return the similarity as well.

        Returns:
            HRR: the nearest neighbour of probe.
            self.dtype: the similarity of probe with the returned trace.
        """
        if showsim:
            return self.nearest(probe, 1)

        return self.nearest(probe, 1)[0][0]

class NaiveMemory(CleanupMemory):
    """Naive cleanup memory; searches by getting the k nearest neighbours with a probe and
    all items.

    Attributes:
        n (int): Number of stored vectors (may change); may differ from reserved memory.
        d (int): Number of dimensions of stored vectors (may not be changed).
        dtype (dtype): Type of the vectors stored in cleanup memory.
        sim (Callable): Function that computes the similarity between two vectors in case they 
                        aren't HRRs. Default cosine similarity.
    """

    def __init__(self, n:int, d:int, 
                 data: HRRArray | None = None,
                 dtype:DTypeLike=np.float64, 
                 sim:Callable[[ArrayLike, ArrayLike], DTypeLike] = lambda u, v: (u@v)/(u@u * v@v),
                 **kwargs
                 ):
        """Initialize an n by d cleanup memory.

        Args:
            n (int): Number of vectors in the cleanup memory.
            d (int): Dimension of vectors to be stored in the memory (must be constant).
            dtype (DTypeLike): numpy dtype for the memory.
        
        Returns:
            None
        """

        super().__init__(0, d, dtype=dtype, sim=sim)
        internal_n = 2**int(np.ceil(np.log2(n)))
        self.memory = HRRArray(internal_n, d, data=np.zeros((internal_n, d), dtype=self.dtype))
        self.populate(data)

    def populate(self,
                 data: np.ndarray | HRRArray):
        """Adds several HRRs to the memory.

        Args:
            data (np.ndarray | HRRArray): HRRs to append.

        Returns:
            None
        """
    
        k = len(data)

        self._grow(k)

        if isinstance(data, HRRArray):
            self.memory.M[self.n:self.n+k, :] = data.M[:, :]
        else:
            self.memory.M[self.n:self.n+k, :] = data[:, :]

        self.n += k

    def _grow(self, k:int):
        """Doubles memory size if n + k > len(memory)

        Args:
            k (int): number of items to be added.

        Returns:
            None
        """

        if self.n + k > len(self.memory):
            internal_n = 2**int(np.ceil(np.log2(self.n + k)))
            newmem = np.zeros((internal_n, self.d), dtype=self.dtype)
            newmem[:self.n, :] = self.memory.M[:self.n, :]
            self.memory = HRRArray(internal_n, self.d, data=newmem)
        
        self.n += k


    def nearest(self, 
                probe:np.ndarray | HRR, 
                k:int=1
                ) -> tuple[HRRArray, np.ndarray]:
        """Returns k nearest neighbours of probe using probe's similarity measure or self.sim 
        function if unavailable (default 1), and similarities of neighbours, in descending order.

        Args:
            probe (np.ndarray | HRR): The vector whose neighbours you want to find.
            k (int): Number of neighbours to find.

        Returns:
            HRRArray: The top k HRRs in the search (greatest similarity first).
        """

        sims = np.zeros(self.n)

        if isinstance(probe, HRR):
            sims[:] = (self.memory == probe)[:self.n]
        else:
            sims[:] = [self.sim(self.memory.M[i, :], probe) for i in range(self.n)]

        sorted_order = np.argpartition(sims[:self.n], -k)[::-1] # permutation that sorts the similarities
        top_items = sorted_order[:k]

        return HRRArray(k, self.d, data=self.memory[top_items]), sims[top_items]

    def insert(self,
               trace: np.ndarray | HRR
               ):
        """Adds a new HRR into the memory, expanding it if necessary.
        
        Args:
            trace (np.ndarray | HRR): the vector to add to memory.

        Returns:
            None
        """
        self._grow(1)

        if isinstance(trace, HRR):
            self.memory.M[self.n, :] = trace.v[:]
        else:
            self.memory.M[self.n, :] = trace[:]

        self.n += 1
    
    def free(self, k:int):
        """Frees k rows of the memory, without zeroing.

        Args:
            k (int): the number of rows to free.

        Returns:
            None
        """

        self.n -= k

class HNSWMemory(CleanupMemory):
    """A hierarchical navigable small world (HNSW) cleanup memory.

    Based on Yalkov & Yashuin (2018), HNSW is a sparse graph over a vector space designed to make 
    approximate nearest neighbour search efficient. When a vector is added, it is connected with
    its top k nearest neighbours in the graph, and, with some constant probability p, it is added
    to another graph on the next "layer" up, repeatedly, until it is no longer on the layer below
    the current one. Search proceeds by traversing the graph to find the nearest neighbour at the
    top level, then starting from the same vertex but on the next level down, repeating the search,
    and so on to the bottom layer. Nearest neighbour search is therefore achieved in O(logn) time,
    although the nearest neighbour is not guaranteed to be exact.

    doi: https://doi.org/10.1109/TPAMI.2018.2889473

    Attributes:
        n (int): the total number of vertices currently at the bottom layer.
        d (int): the dimension of the vectors on the space.
        graph (Graph):  a list of vectors paired with a horizontal adjacency list, and each vertex's 
                        list position on the next layer.
        dtype (dtype): Type of the vectors stored in cleanup memory.
        sim (Callable): Function that computes the similarity between two vectors in case they 
                        aren't HRRs. Default cosine similarity.
    """
    pass


class Graph[n, d, l]:
    """A hierarchical graph over a d-dimensional vector space with n vertices at the bottom layer,
    and l layers.

    Attributes:
    n (int): Number of vertices at the bottom.
    d (int): Number of dimensions of the space the graph lives on.
    l (int): Number of layers of the graph (scales approx. logarithmically with n).
    m (float): Power for random number when setting level of inserted vector.
    max_edges (int): The maximum number of edges per vertex.
    k_nearest (int): The number of vertices to connect to a newly inserted vertex.
    search (Callable): The search algorithm to find the k nearest neighbours of an inserted vector. 
    vertices (ndarray): A matrix containing our vertices. Each row is a vertex.
    edges (list):   A list of lists of indices, specifying each vertex's neighbours on each level.
                    Moving between levels just changes the edge list.
    p_holo (float): The probability that a hologram will be added after 2^(qlog2(n)) insertions. A 
                    hologram averages a sample of k vertices, making it easier to reach more nodes.
    q_holo (float): The number of insertions to use to checkpoint the creation of holograms.
    k_holo (int):   The number of vectors to compose for building a hologram.
    priv (float):   Multiplies with p for holograms, like a hologram "privilege" level.
    """
   
    def __init__(self, d:int, data:np.ndarray | None = None, p:float=0.1, 
                 m:float=0.2, k_nearest:int=10,
                 p_holo:float=0.0, q_holo:float=100, 
                 k_holo:int=10, priv:float=1.0, max_edges:int=50):
        """Instantiates graph with vector vertices and multiple edge lists.

        Args:
            d (int): Dimension of the stored array (will be overridden if data is specified.
            data (np.ndarray | None): Vectors to use as vertices at initialization.
        """

        self.d = d

        if not data:
            self.n = 0
        else:
            self.n, self.d = data.shape # if data shape and d are incongruous, data shape overrides
            self.vertices = data

        self.layers = []
        self.edges = []
        self.m = m
        self.m_conn = 0
        self.k_nearest = k_nearest
        self.p = p
        self.p_holo = p_holo
        self.k_holo = k_holo
        self.priv = priv
        self.max_edges = max_edges

    def _grow(self, k:int):
        """Doubles memory size if n + k > len(memory)

        Args:
            k (int): number of items to be added.

        Returns:
            None
        """

        if self.n + k > len(self.vertices):
            internal_n = 2**int(np.ceil(np.log2(self.n + k)))
            newmem = np.zeros((internal_n, self.d), dtype=np.float64)
            newmem[:self.n, :] = self.vertices[:self.n, :]
            self.vertices = newmem
            for edgelist in self.edges:
                for i in range(self.n, self.n+k):
                    edgelist[i] = []

        self.n += k



    def insert(self, v:np.ndarray):
        """Direct implementation of Malkov & Yashunin (2019, p. 827). Inserts a vertex into an 
        HNSW graph.

        First chooses the highest layer the new vertex will live on. For every layer above that,
        traverses to the nearest neighbour of v. For every layer on that layer and below, appends
        the index of v and connects v with its k nearest neighbours. Prunes edges exceeding the
        maximum number.

        Args:
            v (np.ndarray): The new vector to construct a vertex with.
        
        Returns:
            None
        """

        
        i = self.n # index of the new vertex
        if i == 0:
            self.vertices = np.zeros((1, 256))
            self.n = 1
        else:
            self._grow(1)
        
        self.vertices[i, :] = v
        w = [] # nearest neighbour candidate list
        newv_layer = int(np.floor(-np.log(np.random.random()) * self.m))
        # Make sure all layers have a list to store vertices in
        highest_layer = min(len(self.layers), newv_layer)
        while len(self.layers) <= newv_layer:
            self.layers.append({i}.copy())
            self.edges.append({i: [].copy()})

        ep = list(np.random.choice(list(self.layers[-1]), 
                                   size=min(self.k_nearest, len(self.layers[-1])), 
                                   replace=False
                                   )) # entrypoints
        self.l = len(self.layers)
        # On layers where we aren't placing the new vertex, just traverse down
        for j in range(self.l-1, newv_layer-1, -1):
            w = self.search_layer(v, ep, j, k=1) # gets nearest neighbours
            ep = w

        for j in range(newv_layer, -1, -1):
            if i not in self.layers[j]:
                self.layers[j].add(i)
            if i not in self.edges[j]:
                self.edges[j][i] = []
            l, e = self.layers[j], self.edges[j]
            w = self.search_layer(v, ep, j) # gets nearest neighbour candidates
            neighbours = self.select_neighbours(v, w, j) # get nearest neighbours from candidates
            for neighbour in neighbours: # connect all neighbours to i
                if neighbour != i:
                    if i not in self.edges[j][neighbour]:
                        self.edges[j][neighbour].append(i)
                    if neighbour not in self.edges[j][i]:
                        self.edges[j][i].append(neighbour)
                    self.m_conn += 1
                    
                    if len(self.edges[j][neighbour]) > self.max_edges: # cap number of edges to max
                        self.edges[j][neighbour] = self.select_neighbours(self.vertices[neighbour], 
                                                                          self.edges[j][neighbour],
                                                                          j, k=self.max_edges)
            ep = w
        


    def search_layer(self, v:np.ndarray, ep:list[int], layer:int, k:int|None=None) -> list[int]:
        """Searches a layer for the k nearest neighbours of v.

        Args:
            v (np.ndarray): Probe to search over layer with.
            ep (int): Entrypoint for the search.
            layer (int): Layer the search takes place on.
            k (int | None): Number of neighbours to return.
        
        Returns:
            list[int]: a list of k nearest neighbour indices on selected layer.
        """
        
        # If k is not specified, set to default
        if k is None:
            k = min(len(self.layers[layer]), self.k_nearest)

        visited = ep.copy() # visited nodes
        candidates = ep.copy() # candidates whose neighbours we consider adding to w
        w = ep.copy() # list of nearest neighbours

        #if len(w) < k:
        #    w += list(np.random.choice(self.layers[layer], size=k-len(w), replace=False))

        #candidate_neighbourhood = []
        #for can in candidates:
        #    for neighbour in edges[layer][can]:
        #        if neighbour not in visited:
        #            candidate_neighbourhood.append(neighbour)

        # get similarity of all candidates and current neighbours to v
        sim_cands = [self.vertices[can] @ v for can in candidates]
        sim_ws = [self.vertices[ww] @ v for ww in w]
        
        while len(candidates) > 0: 
            c = np.argmax(sim_cands) # find the closest of all candidate neighbours
            sim_c = sim_cands.pop(c) # keep value, but remove it from later consideration
            cand = candidates.pop(c)
            farthest_w = np.argmin(sim_ws) # find the farthest of all neighbours

            # if the nearest candidate is farther than the farthest neighbour, we're done
            if sim_c < sim_ws[farthest_w]:
                break
            
            # Otherwise, search the candidate's neighbourhood 
            for neighbour in self.edges[layer][cand]:
                if neighbour not in visited:
                    visited.append(neighbour) # add it to the visited list so we only check it once
                    farthest_w = np.argmin(sim_ws) # get the farthest w

                    # If neighbour is better than the worst current neighbour, or |w| < k, add it
                    if self.vertices[neighbour] @ v > sim_ws[farthest_w] or len(w) < k:
                        sim = v @ self.vertices[neighbour]
                        candidates.append(neighbour)
                        sim_cands.append(sim)
                        w.append(neighbour)
                        sim_ws.append(sim)
                        if len(w) > k:
                            del w[farthest_w]
                            del sim_ws[farthest_w]

        return w

    def select_neighbours(self, v:np.ndarray, candidates:list[int], layer:int, k:int|None=None,
                          extend:bool=True, keep:bool=True) -> list[int]:
        if k is None:
            k = self.k_nearest

        r = []
        r_sims = []
        sims = [v @ self.vertices[c] for c in candidates] # similarity of each candidate to v
        w = candidates.copy() 
        if extend:
            for c in candidates:
                for conn in self.edges[layer][c]:
                    if conn not in w:
                        w.append(conn)
                        sims.append(v @ self.vertices[conn])
        discard = []
        discard_sims = []
        while len(w) > 0 and len(r) < k:
            nearest = np.argmax(sims)
            sim_e = sims.pop(nearest)
            e = w.pop(nearest)

            if not r_sims or np.max(r_sims) < sim_e:
                r.append(e)
                r_sims.append(sim_e)
            else:
                discard.append(e)
                discard_sims.append(e)

        if keep:
            while len(discard) > 0 and len(r) < k:
                nearest = np.argmax(discard_sims)
                r_sims.append(discard_sims.pop(nearest))
                r.append(discard.pop(nearest))

        return sorted(r, key=lambda i: r_sims[r.index(i)], reverse=True)
