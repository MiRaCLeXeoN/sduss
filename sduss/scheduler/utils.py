from typing import Iterable

def find_gcd(resolutions: Iterable[int]):

    def euclid(x, y):
        while y:
            x, y = y, x % y
        return x

    cur_gcd = None
    for res in resolutions:
        if cur_gcd is None:
            cur_gcd = res
        else:
            cur_gcd = euclid(cur_gcd, res)
    return cur_gcd