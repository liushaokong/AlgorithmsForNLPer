"""
here is to show the 4 Binary Tree (BT) traversals.
    1. preorder
    2. inorder 
    3. postorder
    4. level order
"""
import collections

class TreeNode():
    def __init__(self, val):
        self.val = val
        self.left = None 
        self. right = None
    
    def preorder(self, root):
        if not root:
            return []  # should be [] instead of None for add operation
        return [root.val] + self.preorder(root.left) + self.preorder(root.right)
    
    def inorder(self, root):
        if not root:
            return []
        return self.inorder(root.left) + [root.val] + self.inorder(root.right)
    
    def postorder(self, root):
        if not root:
            return []
        return self.postorder(root.left) + self.postorder(root.right) + [root.val]
    
    def levelorder(self, root):
        """
        a classic BFS method
        """
        if not root:
            return []
        
        result = []
        queue = collections.deque()
        queue.append(root)

        # visited = set([root])  # set an iterable objec

        while queue:
            level_size = len(queue)  # process layer by layer, record for level infomation.
            level_result = []  # current level container

            for _ in range(level_size):
                # generate current node
                node = queue.popleft()

                # if node not in visited:
                #     visited.append(node)

                # process current node
                level_result.append(node.val)
                
                # generate next nodes
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # process whole level result
            result.append(level_result)
        return result
