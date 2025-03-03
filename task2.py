import timeit
import matplotlib.pyplot as plt
from functools import lru_cache


# Реалізація функції для обчислення чисел Фібоначчі з використанням LRU-кешу
@lru_cache(maxsize=None)
def fibonacci_lru(n):
    if n < 2:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)


# Реалізація Splay Tree
class SplayTreeNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None


class SplayTree:
    def __init__(self):
        self.root = None

    def _splay(self, root, key):
        if root is None or root.key == key:
            return root

        if root.key > key:
            if root.left is None:
                return root
            if root.left.key > key:
                root.left.left = self._splay(root.left.left, key)
                root = self._rotate_right(root)
            elif root.left.key < key:
                root.left.right = self._splay(root.left.right, key)
                if root.left.right is not None:
                    root.left = self._rotate_left(root.left)
            return self._rotate_right(root) if root.left is not None else root
        else:
            if root.right is None:
                return root
            if root.right.key > key:
                root.right.left = self._splay(root.right.left, key)
                if root.right.left is not None:
                    root.right = self._rotate_right(root.right)
            elif root.right.key < key:
                root.right.right = self._splay(root.right.right, key)
                root = self._rotate_left(root)
            return self._rotate_left(root) if root.right is not None else root

    def _rotate_right(self, root):
        new_root = root.left
        root.left = new_root.right
        new_root.right = root
        return new_root

    def _rotate_left(self, root):
        new_root = root.right
        root.right = new_root.left
        new_root.left = root
        return new_root

    def insert(self, key, value):
        if self.root is None:
            self.root = SplayTreeNode(key, value)
            return
        self.root = self._splay(self.root, key)
        if self.root.key == key:
            self.root.value = value
            return
        new_node = SplayTreeNode(key, value)
        if self.root.key > key:
            new_node.right = self.root
            new_node.left = self.root.left
            self.root.left = None
        else:
            new_node.left = self.root
            new_node.right = self.root.right
            self.root.right = None
        self.root = new_node

    def search(self, key):
        self.root = self._splay(self.root, key)
        if self.root and self.root.key == key:
            return self.root.value
        return None


# Реалізація функції для обчислення чисел Фібоначчі з використанням Splay Tree
def fibonacci_splay(n, tree):
    if n < 2:
        return n
    if (result := tree.search(n)) is not None:
        return result
    result = fibonacci_splay(n - 1, tree) + fibonacci_splay(n - 2, tree)
    tree.insert(n, result)
    return result


# Вимірювання часу виконання
def measure_time(func, *args):
    start_time = timeit.default_timer()
    func(*args)
    return timeit.default_timer() - start_time


# Порівняння продуктивності
n_values = list(range(0, 951, 50))
lru_times = []
splay_times = []

for n in n_values:
    lru_time = measure_time(fibonacci_lru, n)
    lru_times.append(lru_time)

    splay_tree = SplayTree()
    splay_time = measure_time(fibonacci_splay, n, splay_tree)
    splay_times.append(splay_time)

# Побудова графіка
plt.plot(n_values, lru_times, label="LRU Cache")
plt.plot(n_values, splay_times, label="Splay Tree")
plt.xlabel("n")
plt.ylabel("Time (s)")
plt.title("Performance Comparison of Fibonacci Calculation")
plt.legend()
plt.show()

# Виведення таблиці результатів
print(f"{'n':<10}{'LRU Cache Time (s)':<20}{'Splay Tree Time (s)':<20}")
print("-" * 50)
for n, lru_time, splay_time in zip(n_values, lru_times, splay_times):
    print(f"{n:<10}{lru_time:<20.10f}{splay_time:<20.10f}")
