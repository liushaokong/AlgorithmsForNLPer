"""
to show some classic problems could be solved using dynamic programming.
the key is to define dp array.

if you would like to remember 2 dp problems, I believe it should be coin_change and edit_distance.
one is a number problem while the other is a string one.
"""

def coin_change(coins, amount):
    """
    use least number coins combination to get the target amount.
    """
    Max = amount + 1

    # define and initialize dp
    dp = [Max for _ in range(Max)]  # dp[i] could be any number larger than amount
    dp[0] = 0

    for i in range(1, Max):  # amount array
        for j in range(len(coins)):  # for coin in coins
            if coins[j] <= i:
                dp[i] = min(dp[i], dp[i-coins[j]] + 1)
    
    if dp[amount] > amount:
        return -1
    return dp[amount]


def min_triangle_path_sum(triangle):
    """
    from bottom to top
    """
    if not triangle:
        return 0

    # define and init dp 
    dp = triangle[-1]  # the last row
    length = len(triangle)

    for i in range(length - 2, -1, -1):  # from the 2nd last to the top
        for j in range(len(triangle[i])):
            dp[j] = min(dp[j], dp[j+1]) + triangle[i][j]  

    return dp[0]


def edit_distance(word1, word2):
    """
    use a 2-d dp array, dp[i][j] = edit_distance(word[:i][:j])
    dp[i][j] for word1[i-1], word2[j-1]
    """
    m, n = len(word1), len(word2)

    # define dp
    dp = [[0 for _ in (n+1)] for _ in (m+1)]  # {rows: m+1, col: n+1}

    for i in range(m+1):  # word2 = ""
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(m+1):
        for j in range(n+1):
            dp[i][j] = min(
                dp[i-1][j-1] + (0 if word1[i-1] == word2[j-1] else 1),
                dp[i-1][j],
                dp[i][j-1]
            )
    return dp[m][n]




