"""
examples of how the binary search and newton method is realized.

the idea if the Newton method is as following:
f'(x) for getting the point x0 whree f(x0) = 0
f"(x) for getting the point x0 where f'(x0) = 0

or in Chinese:
一阶导数求零点，二阶导数求极值。

here is to show using f'(x) to get the zero point.

a line with a point (x0, f(x0)) could be described as 
f(x) = f'(x0)(x - x0) + f(x0)
let f(x1) = 0, it could be get x1 = x0 - f(x0) / f'(x0)

similarly, a curve 
    g(r) = r^2 - x, where x is a constant here, could be dexcribed as 
    g(r) = 2r(x - r0) + g(r0), within a small range near r0 (try to convince yourself),
let g(r) = 0, it could be get
    r1 = r0 - g(r0) / 2r0, or 
    r1 = r0 - (r0^2 - x) / 2r0 = (r0 + x/ r0)
"""

def binary_search(array, target):
    """
    arrray: a sorted array
    """
    left, right = 0, len(array)

    while left <= right:
        mid = left + (right - left) / 2  # (left + right) / 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


class sqrt:
    """
    consider only int here.
    """
    def binary_search(self, x):
        if x <= 1:
            return x
        
        left, right = 1, x
        while left <= right:
            mid = left + (right - left) / 2
            square = mid * mid
            if square == x:
                return mid
            elif square > x:
                right = mid - 1 
            else:
                left = mid + 1
                res = mid
        return res

    def newton_method(self, x):
        r = x
        while r * r > x:
            r = (r + x / r) / 2
        return r

