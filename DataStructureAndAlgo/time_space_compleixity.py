"""
take Fibonacci for example to show time and space complexity optimization.
"""

import numpy as np

# recursive
def fib1(n):
    """
    time complextity: O(2^n)
    space complexity: O(2^n)
    """
    if n < 2:
        return 1
    return fib1(n-1) + fib1(n-2)

# time omtimization
def fib2(n):
    """
    the most easiest dp method
    time complexity: O(n)
    space complexity: O(n)
    """
    # dp = [1 for _ in range(n+1)]  # define dp and set initial state at the same time
    dp = [1] * (n + 1)
    
    for i in range(2, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]  # or return dp

# space optimization
def fib3(n):
    """
    using rolling array to save space
    time complexity: O(n)
    space complexity: O(1)
    """
    a0 = 1
    a1 = 1
    
    for i in range(2, n+1):
        a1, a0 = a0 + a1, a1
    
    return a1

# use matirx to speed up
def matrix_power(matrix, n):
    """
    to cal matirx power
    """
    matpow = np.eye(2)
    
    while n:
        
        if n & 1:
            matpow = np.matmul(matpow, matrix)
        
        matrix = np.matmul(matrix, matrix)
        
        n >>= 1
    
    return matpow

def fib4(n):
    """
    time complexity: O(log-n)
    space complexity: O(1)

    f(n)   = [1, 1] f(n-1) 
    f(n-1) = [1, 0] f(n-2)
    """
    matrix = np.array([1, 1, 1, 0]).reshape(2, 2)
    
    return matrix_power(matrix, n)[0][0]  


if __name__ == "__main__":
    dp = fib2(12)
    print(dp)