from typing import List


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

