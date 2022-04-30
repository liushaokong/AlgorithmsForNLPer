"""
leetcode 64. minimum path sum from top-left to bottom_right
here to show how to get the minimum path sum of a matrix, and get the path
"""

class Solution:
    """
    minPathSum returns only the min_path_sum
    minPathSum2 returns min_path_sum and path
    """
    def minPathSum(self, grid):
        if not grid:
            return 
        
        # {r: row, c: column}
        r, c = len(grid), len(grid[0])
        
        # step 1: define dp 
        dp = [[0 for _ in range(c)] for _ in range(r)]

        # step 2: initialize dp
        dp[0][0] = grid[0][0]
        
        for i in range(1, r):  # for colum[0]
            dp[i][0] = dp[i-1][0] + grid[i][0]  # accumulate
            
        for i in range(1, c):  # for row[0]
            dp[0][i] = dp[0][i-1] + grid[0][i]
        
        # step 3: iterate dp 
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

        return dp[-1][-1]
    
    
    def minPathSum2(self, grid):
        if not grid:
            return 
        
        r, c = len(grid), len(grid[0])
        dp = [[0 for _ in range(c)] for _ in range(r)]
        dp[0][0] = grid[0][0]
        
        # path matrix, use an extra matrix to record path
        path = [[0 for _ in range(c)] for _ in range(r)]
        path[0][0] = 0
        
        for i in range(1, r):  # for column[0]
            dp[i][0] = dp[i-1][0] + grid[i][0]
            path[i][0] = 0  # only can be from up
        
        for j in range(1, c):  # for row[0]
            dp[0][j] = dp[0][j-1] + grid[0][j]
            path[0][j] = 1  # only can be from left
        
        direction = 0
        for i in range(1, r):
            for j in range(1, c):
                if dp[i][j-1] < dp[i-1][j]:  # move rignt
                    direction = 1
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                else:
                    direction = 0  # moving down
                    dp[i][j] = dp[i-1][j] + grid[i][j]
                
                path[i][j] = direction
                    
        return dp[-1][-1], path


if __name__ == "__main__":
    matrix = [[1,  3, 1],
              [16, 5, 18],
              [4,  2, 81]]
    min_sum, path = Solution().minPathSum2(matrix)
    print(min_sum)
    print(path)

    r = len(path)-1
    c = len(path[0])-1

    path_num = [matrix[-1][-1]]

    while r >= 0 and c > 0:
        if path[r][c] == 0:
            r -= 1
        else:
            c -= 1   
        path_num.append(matrix[r][c])
        
    print(path_num)