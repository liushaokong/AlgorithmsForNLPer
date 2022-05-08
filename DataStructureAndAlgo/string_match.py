"""
some famous string (matching) algorithms, like 
KMP, 

manacher, 
trie_tree
"""

def kmp(main, pattern):
    """
    main: main string
    pattern: 
    """
    n, m = len(main), len(pattern)

    if m == 0:
        return 0 
    if n <= m:
        return 0 if main == pattern else -1 
    
    next = get_next(pattern)
    j = 0  # for current matching position in pattern
    
    for i in range(n):  # n = len(main)
        while j > 0 and main[i] != pattern[j]:  # encounter a bad char in main
            j = next[j-1] + 1  # reset matching position
            """
            the key idea here is once a char not matched, 
            using the special structure of pattern,
            move as many steps as possible,
            use the array calculated to find out how many steps the pattern could move.

            pattern: ababac
                          j
            pattern[j] not match main[i], and pattern[:j] matched ababa
            next[j-1] = 2, ending position for aba,
            pattern move 2 steps and start matching pattern[3]
            """
        
        if main[i] == pattern[j]:
            if j == m-1:  # all pattern chars matched 
                return i - m + 1
            else:  # one more char matched
                j += 1
    return -1
    
def get_next(pattern):
    """
    get next array for the kmp string match
    ababa
    """
    m = len(pattern)
    next = [-1] * m 

    for i in range(1, m):
        j = next[i-1]  # already matched prefix ending postion

        while j != -1 and pattern[j+1] != pattern[i]:
            j = next[j]  # find out j using next, 
            # think about the j -= 1 in the loop for understanding j = next[j]
        
        if pattern[j+1] == pattern[i]:  # add one more char to matched prefix
            j += 1
        
        next[i] = j
    
    return next

def test_kmp():
    m_str = "aabbbbaaabbababbabbbabaaabb"
    p_str = "abbabbbabaa"

    print('--- search ---')
    print('[Built-in Functions] result:', m_str.find(p_str))
    print('[kmp] result:', kmp(m_str, p_str))


if __name__ == '__main__':
    test_kmp()
