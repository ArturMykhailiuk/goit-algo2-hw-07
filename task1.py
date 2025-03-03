import random
import time
from functools import lru_cache


# Функція для обчислення суми елементів на відрізку без використання кешу
def range_sum_no_cache(array, L, R):
    return sum(array[L : R + 1])


# Функція для оновлення значення елемента масиву без використання кешу
def update_no_cache(array, index, value):
    array[index] = value


# Зберігаємо ключі кешу для подальшого очищення
cached_ranges = set()


# Функція для обчислення суми елементів на відрізку з використанням LRU-кешу
@lru_cache(maxsize=1000)
def range_sum_with_cache(array, L, R):
    # print(f"Запит на обчислення суми елементів на відрізку {L}..{R}")
    # print(f"Отримана сума: {sum(array[L : R + 1])}")
    cached_ranges.add((L, R))
    return sum(array[L : R + 1])


# Функція для оновлення значення елемента масиву з використанням кешу
def update_with_cache(array, index, value):
    array[index] = value
    # Очищення кешу тільки для відповідних діапазонів
    keys_to_clear = [key for key in cached_ranges if key[0] <= index <= key[1]]
    for key in keys_to_clear:
        # print(f"Очищення кешу для діапазону {key}")
        range_sum_with_cache.cache_clear()
        cached_ranges.remove(key)


# Генерація масиву та запитів
N = 100000
Q = 50000
array = [random.randint(1, 100) for _ in range(N)]  # Зберігаємо масив як список
queries = [
    (
        ("Range", L := random.randint(0, N - 1), random.randint(L, N - 1))
        if random.random() < 0.5
        else ("Update", random.randint(0, N - 1), random.randint(1, 100))
    )
    for _ in range(Q)
]

# Виконання запитів без кешу
start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_no_cache(array, query[1], query[2])
    elif query[0] == "Update":
        update_no_cache(array, query[1], query[2])
no_cache_time = time.time() - start_time

# Виконання запитів з кешем
start_time = time.time()
for query in queries:
    if query[0] == "Range":
        range_sum_with_cache(tuple(array), query[1], query[2])
    elif query[0] == "Update":
        update_with_cache(array, query[1], query[2])
cache_time = time.time() - start_time

# Виведення результатів
print(f"Час виконання без кешування: {no_cache_time:.2f} секунд")
print(f"Час виконання з LRU-кешем: {cache_time:.2f} секунд")

print(range_sum_with_cache.cache_info())
