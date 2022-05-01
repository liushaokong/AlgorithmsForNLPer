"""
there are several sort methods, here is to show 2 most commmly used, 
namely, {quick_sort, merge_sort}
"""
from random import randint 


class QuickSort:
    """
    methods_0: the basic mtthod,
    method_1: randomly choose pivot


    I believe method_0 and method_1 are enough for you to prepare your job interview.
    If you are ambitious, move on for the others.
    """
    def method_0(self, a):
        """
        use the first item as pivot
        a: an array list
        """
        if len(a) <= 1:  # len(a) < 2
            return a
        
        pivot = a[0]
        a = a[1:]
        smaller = [n for n in a if n <= pivot]
        bigger = [n for n in a if n > pivot]

        return self.method_0(smaller) + [pivot] + self.method_0(bigger)
    
    def method_1(self, a):
        """
        randomly select pivot
        """
        if len(a) <= 1:
            return a 
        
        pivot = a[randint(0, len(a) - 1)]
        smaller, equal, bigger = [], [], []

        for n in a:  # only once
            if n < pivot:
                smaller.append(n)
            if n > pivot:
                bigger.append(n)
            else:
                equal.append(n)
        return self.method_1(smaller) + equal + self.method_1(bigger)
    
    def partition_3(self, A, lo, hi):
        """
        divide A into 2 parts, 
        return the pivot position
        """
        pivot = A[hi]
        i = lo
        for j in range(lo, hi):
            if A[j] < pivot:
                A[i], A[j] = A[j], A[i]
                i += 1
        A[i], A[hi] = A[hi], A[i]
        return i
    def method_3(self, A, lo, hi):
        if lo < hi:
            p = self.partition(A, lo, hi)
            self.quick_sort(A, lo, p-1)
            self.quick_sort(A, p+1, hi)
    

    def partition_4(self, A, lo, hi):
        """
        Hoare partition scheme
        """
        pivot = A[(lo + hi) // 2]
        i = lo - 1
        j = hi + 1
        while True:
            i += 1  # after exchange, move pointers
            j -= 1
            
            while A[i] < pivot:
                i += 1
            
            while A[j] > pivot:
                j -= 1
                
            if i >= j:
                return j
            
            A[i], A[j] = A[j], A[i]
    def method_4(self, A, lo, hi):
        if lo < hi:
            p = self.partition(A, lo, hi)
            self.quick_sort(A, lo, p)
            self.quick_sort(A, p+1, hi)
    

class MergeSort:
    """
    2 parts: merge and merge_sort
    classic method for divide and conquer.
    """
    def merge(self, a, b):
        # merge 2 sorted list
        c = []
        pa, pb = 0, 0
        la, lb = len(a), len(b)

        while pa < la and pb < lb:
            if a[pa] < b[pb]:
                c.append(a[pb])
                pa += 1
            else:
                c.append(b[pb])
                pb += 1
        
        if pa == la:  # merged all items in a
            c.extend(b[pb:])
        if pb == lb:
            c.extend(a[pa:])
        return c
    
    def sort(self, a):
        if len(a) <= 1:
            return a
        
        la = len(a)
        left = self.sort(a[:la // 2])
        right = self.sort(a[la // 2:])
        res = self.merge(left, right)

        return res