"""
here are examples showing how the UnionFind structure method \
is defined and used.

examples are: 
    1. number of islands
    2. friend cycles, almost the same as islands.
"""
class UnionFind():
    def __init__(self, grid):

        m, n = len(grid), len(grid[0])

        # count, parent, rank
        self.count = 0  # all items count 
        self.parent = [-1] * (m * n)
        self.rank = [0] * (m * n)

        # set parent and rank
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    self.parent[i * n + j] = i * n + j  # parent[i] = i
                    self.count += 1

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]  # recursive end condition
    
    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            if self.rank[rootx] > self.rank[rooty]:
                self.parent[rooty] = rootx 
            elif self.rank[rootx] < self.rank[rooty]:
                self.parent[rootx] = rooty
            else:
                self.parent[rooty] = rootx 
                self.rank[rootx] += 1
            self.count -= 1


class Solution:
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0 
        
        uf = UnionFind(grid)
        directions = [
            [0, 1],  # right
            [0, -1],  # left
            [-1, 0],  # down
            [1, 0]  # up
        ]
        m, n = len(grid), len(grid[0])

        for i in range(m):
            for j in range(n):
                if grid[i][j] == "0":
                    continue
                for d in directions:
                    nr, nc = i + d[0], j + d[1]  # r for row and c for column
                    if nr >= 0 and nc >= 0 and nr < m and nc < n and grid[nr][nc] == "1":  # accuire around
                        uf.union(i * n + j, nr * n + nc)
        return uf.count 

