from sortedcontainers import SortedKeyList

class SortedValueDict:
    """
    A dictionary-like data structure that allows 
        for efficient extraction of the minimum value, as well as insertion and removal of key-value pairs.
    Expects (key, value) pairs to be inserted.
    """
    def __init__(self):
        self.value_dict = {}  # Key -> Value mapping
        self.sorted_list = SortedKeyList([], key=lambda x: x[1])  # Sort by value

    def __call__(self):
        return self.value_dict  # Calling the object returns the dictionary

    def insert(self, key, value):
        if key in self.value_dict:
            self.remove_by_key(key)  # Remove old value before inserting a new one
        self.value_dict[key] = value
        self.sorted_list.add((key, value))  # O(log n)

    def remove_by_key(self, key):
        if key in self.value_dict:
            value = self.value_dict.pop(key)
            self.sorted_list.remove((key, value))  # O(log n)

    def extract_min(self):
        if self.sorted_list:
            key, value = self.sorted_list.pop(0)  # O(1)
            del self.value_dict[key]
            return key, value
        return (None, None), None