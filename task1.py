import random
import time
import pandas as pd
from functools import lru_cache

# Ініціалізація параметрів
N = 1000000  # Розмір масиву
Q = 5000  # Кількість запитів
array = [random.randint(1, 100) for _ in range(N)]  # Генерація масиву

# Генерація запитів (50% - сума відрізку, 50% - оновлення)
queries = [
    (
        ("Range", L := random.randint(0, N - 1), random.randint(L, N - 1))
        if random.random() < 0.5
        else ("Update", random.randint(0, N - 1), random.randint(1, 100))
    )
    for _ in range(Q)
]


### 🔹 1. Без кешування
def range_sum_no_cache(L, R):
    return sum(array[L : R + 1])


def update_no_cache(index, value):
    array[index] = value


start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_no_cache(query[1], query[2])
    elif query[0] == "Update":
        update_no_cache(query[1], query[2])
no_cache_time = time.time() - start_time


### 🔹 2. З оптимізованим кешем (LRU Cache)
# Оптимізований підхід: Використовуємо словник для відстеження залежностей
# Для кожного індексу зберігаємо список діапазонів, на які він впливає
index_to_ranges = {i: set() for i in range(N)}


@lru_cache(maxsize=10000)
def range_sum_with_optimized_cache(L, R):
    # Реєструємо залежність: кожен індекс у діапазоні [L, R] впливає на цей кеш
    for i in range(L, R + 1):
        index_to_ranges[i].add((L, R))
    return sum(array[L : R + 1])


def update_with_optimized_cache(index, value):
    array[index] = value
    # Знаходимо тільки ті діапазони, на які впливає цей індекс
    affected_ranges = index_to_ranges[index].copy()
    # Очищаємо кеш тільки для цих діапазонів
    for L, R in affected_ranges:
        range_sum_with_optimized_cache.cache_clear()  # Тут краще було б очистити конкретний ключ, але lru_cache не має такої функції
        # Видаляємо діапазон з усіх індексів, які він містить
        for i in range(L, R + 1):
            index_to_ranges[i].discard((L, R))


start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_with_optimized_cache(query[1], query[2])
    elif query[0] == "Update":
        update_with_optimized_cache(query[1], query[2])
optimized_cache_time = time.time() - start_time


### 🔹 3. З модифікованим кешем (без залежностей)
# Ще одна оптимізація: використовуємо InvalidationCounter
invalidation_counter = 0


@lru_cache(maxsize=10000)
def range_sum_with_counter_cache(L, R, counter):
    # Ігноруємо counter в обчисленнях, він потрібен лише для інвалідації кешу
    return sum(array[L : R + 1])


def update_with_counter_cache(index, value):
    global invalidation_counter
    array[index] = value
    # Просто збільшуємо лічильник при кожному оновленні
    invalidation_counter += 1


start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_with_counter_cache(query[1], query[2], invalidation_counter)
    elif query[0] == "Update":
        update_with_counter_cache(query[1], query[2])
counter_cache_time = time.time() - start_time


### 🔹 4. Segment Tree (для порівняння)
class SegmentTree:
    def __init__(self, array):
        self.n = len(array)
        self.tree = [0] * (4 * self.n)
        self.build(array, 0, 0, self.n - 1)

    def build(self, array, node, start, end):
        if start == end:
            self.tree[node] = array[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            self.build(array, left_child, start, mid)
            self.build(array, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def update(self, idx, value, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if start == end:
            self.tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            if idx <= mid:
                self.update(idx, value, left_child, start, mid)
            else:
                self.update(idx, value, right_child, mid + 1, end)
            self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, L, R, node=0, start=0, end=None):
        if end is None:
            end = self.n - 1
        if R < start or L > end:
            return 0
        if L <= start and end <= R:
            return self.tree[node]
        mid = (start + end) // 2
        left_sum = self.query(L, R, 2 * node + 1, start, mid)
        right_sum = self.query(L, R, 2 * node + 2, mid + 1, end)
        return left_sum + right_sum


segment_tree = SegmentTree(array)

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        segment_tree.query(query[1], query[2])
    elif query[0] == "Update":
        segment_tree.update(query[1], query[2])
segment_tree_time = time.time() - start_time


### 🔹 5. Fenwick Tree (для порівняння)
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def build(self, array):
        for i in range(1, self.n + 1):
            self.add(i, array[i - 1])

    def add(self, index, value):
        while index <= self.n:
            self.tree[index] += value
            index += index & -index

    def prefix_sum(self, index):
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def query(self, L, R):
        return self.prefix_sum(R + 1) - self.prefix_sum(L)

    def update(self, index, value):
        diff = value - (self.query(index, index))
        self.add(index + 1, diff)


fenwick_tree = FenwickTree(N)
fenwick_tree.build(array)

start_time = time.time()
for query in queries:
    if query[0] == "Range":
        fenwick_tree.query(query[1], query[2])
    elif query[0] == "Update":
        fenwick_tree.update(query[1], query[2])
fenwick_tree_time = time.time() - start_time


### 📊 **Виведення результатів у гарному форматі**
data = {
    "Метод": [
        "Без кешу",
        "Оптимізований кеш",
        "Кеш з лічильником",
        "Segment Tree",
        "Fenwick Tree",
    ],
    "Час виконання (сек)": [
        no_cache_time,
        optimized_cache_time,
        counter_cache_time,
        segment_tree_time,
        fenwick_tree_time,
    ],
    "Відносна швидкість": [
        "1.0x",
        f"{no_cache_time/optimized_cache_time:.2f}x",
        f"{no_cache_time/counter_cache_time:.2f}x",
        f"{no_cache_time/segment_tree_time:.2f}x",
        f"{no_cache_time/fenwick_tree_time:.2f}x",
    ],
    "Складність оновлення": [
        "O(1)",
        "O(1) + частково інвалідація",
        "O(1)",
        "O(log N)",
        "O(log N)",
    ],
    "Складність запиту": [
        "O(R - L)",
        "O(1) (кеш)",
        "O(1) (кеш)",
        "O(log N)",
        "O(log N)",
    ],
}

df = pd.DataFrame(data)

# Виводимо красиву таблицю з результатами в консоль
print("\n" + "=" * 80)
print(" " * 30 + "📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ 📊")
print("=" * 80)
print(f"Розмір масиву: {N:,} елементів")
print(f"Кількість запитів: {Q:,} (приблизно {Q//2:,} запитів суми і {Q//2:,} оновлень)")
print("-" * 80)

# Форматування даних таблиці
table_data = []
fastest_time = min(
    no_cache_time,
    optimized_cache_time,
    counter_cache_time,
    segment_tree_time,
    fenwick_tree_time,
)
for i, method in enumerate(data["Метод"]):
    time_value = data["Час виконання (сек)"][i]
    rel_speed = data["Відносна швидкість"][i]
    time_str = f"{time_value:.4f} сек"

    # Додаємо позначку найшвидшого методу
    if time_value == fastest_time:
        time_str += " 🏆"

    table_data.append(
        [
            method,
            time_str,
            rel_speed,
            data["Складність оновлення"][i],
            data["Складність запиту"][i],
        ]
    )


from tabulate import tabulate

headers = [
    "Метод",
    "Час виконання",
    "Прискорення",
    "Складність оновлення",
    "Складність запиту",
]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

print("-" * 80)
