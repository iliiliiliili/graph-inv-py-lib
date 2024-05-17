import numpy as np


class MaxHeap:

    def __init__(self) -> None:
        pass

    # def set_context(self, keys: np.ndarray, values: np.ndarray)

    def push(self, key, value, keys: np.ndarray, values: np.ndarray, size):
        keys[size[0]] = key
        values[size[0]] = value

        size += 1

        self.siftdown(0, size[0] - 1, key, value, keys, values)

    def pop(self, keys: np.ndarray, values: np.ndarray, size):

        latest_id = size - 1
        size -= 1

        if latest_id > 0:
            self.assign(0, latest_id, keys, values)
            self.siftup(0, size, self.top_key(keys), self.top_value(values), keys, values)

    def top_value(self, values):
        return values[0]

    def top_key(self, keys):
        return keys[0]

    def siftdown(
        self, start, position, new_key, new_value, keys: np.ndarray, values: np.ndarray
    ):

        while position > start:
            parent = (position - 1) >> 1
            parent_value = values[parent]

            if new_value > parent_value:
                self.assign(position, parent, keys, values)
                position = parent
            else:
                break

        self.assign_value(position, new_key, new_value, keys, values)

    def siftup(self, position, end, new_key, new_value, keys, values):
        start = position

        child_position = 2 * position + 1

        while child_position < end:
            right_child_position = child_position + 1

            if (right_child_position < end) and not (values[child_position] > values[right_child_position]):
                child_position = right_child_position

            self.assign(position, child_position, keys, values)
            position = child_position
            child_position = 2 * position + 1

        self.assign_value(position, new_key, new_value, keys, values)
        self.siftdown(start, position, new_key, new_value, keys, values)

    def assign(self, target, source, keys, values):
        keys[target] = keys[source]
        values[target] = values[source]

    def assign_value(self, target, key, value, keys, values):
        keys[target] = key
        values[target] = value
