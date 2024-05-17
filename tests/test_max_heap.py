import numpy as np
import unittest 

from src.ginv.max_heap import MaxHeap


class TestMaxHeap(unittest.TestCase):

    def test_max_sorted_pop(self):

        inputs = [
            (14, 4.0),
            (11, 1.0),
            (13, 3.0),
            (10, 0.0),
            (12, 2.0),
        ]

        targets = [
            (14, 4.0),
            (13, 3.0),
            (12, 2.0),
            (11, 1.0),
            (10, 0.0),
        ]

        results = []

        keys = np.empty([10], dtype=np.int32)
        values = np.empty([10], dtype=np.float32)
        size = np.zeros([1], dtype=np.int32)

        max_heap = MaxHeap()

        for (k, v) in inputs:
            max_heap.push(k, v, keys, values, size[0:1])

        while size[0:1] > 0:
            results.append((max_heap.top_key(keys), max_heap.top_value(values)))
            max_heap.pop(keys, values, size[0:1])
        
        self.assertEqual(len(targets), len(results))
        self.assertEqual(results, targets)
