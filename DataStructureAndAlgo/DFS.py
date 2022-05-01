"""
here are some examples showing how the DFS work.
"""

import collections

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None 
        self. right = None


class Traversal:
    """
    to show using DFS to get all the values of of Binary Tree by level,
    please compare with the method used in BT_traversal.py
    """
    def level_order_traversal(self, root):
        if not root:
            return []
        
        self.result = []  # use a result to store global result
        self._dfs(node=root, level=0)  # start with current node and level
        return self.result

    def _dfs(self, node, level):
        """
        create a list for each level
        """
        if not node:
            return
        
        if len(self.result) < level + 1:  # [0, 1, ..., level]
            self.result.append([])
        
        self.result[level].append(node.val)  # add current node 

        self._dfs(node.left, level + 1)  # add the children's of current node to the next level.
        self._dfs(node.right, level + 1)


class DepthOfBinaryTree:
    def max_depth(self, root):
        if not root:
            return 0
        return 1 + max(self.max_depth(root.left),
                       self.max_depth(self.right))
    
    def min_depth(self, root):
        if not root:
            return 0
        left_min = self.min_depth(root.left)
        right_min = self.min_depth(root.right)

        if left_min == 0 or right_min == 0:  # left_min * right_min == 0
            return left_min + right_min + 1

        return 1 + min(self.min_depth(root.left),
                       self.min_depth(root.right))


class GenerateParentheses:
    """
    leetcode 22,
    also a graph problem which could be solved using DFS.
    """
    def generate(self, n):
        self.results = []  # to store all the results
        self._dfs(0, 0, n, result="")
        return self.results

    def _dfs(self, left, right, n, result):
        if left == n and right == n:  # end condition, or recursion terminator
            self.results.append(result)
            return
        if left < n:
            self._dfs(left + 1, right, n, result + "(")
        if left > right and right < n:  # left shold be larger than right
            self._dfs(left, right + 1, n, result + ")")


class NQueens:
    """
    Even Gauss did not solve eight queens puzzle.
    """
    def solve(self, n):
        self.results = []  # global results
        self._dfs([], n, [], [])
        return self.results

        # return len(self.result) if number of solutions is needed

        # if need to print all the results,
        # return [["." * i + "Q" + "." * (n - i -1) for i in result] for result in self.results] 

    def _dfs(self, queens, n, xy_diff, xy_sum):
        """
        queens: temp results, a list to store row positions
        """
        row = len(queens)
        if row == n:  # find all postions of 1 solution
            self.results.append(queens)
            return None
    
        for col in range(n):  # get a column position for each row
            if col not in queens and \
               row - col not in xy_diff and \
               row + col not in xy_sum:
                self._dfs(queens + [col], n, xy_diff + [row -col], xy_sum + [row + col])
      

if __name__ == "__main__":
    results = NQueens().solve(8)
    print(results)  # 92
