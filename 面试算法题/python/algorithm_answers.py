"""Original Python reference implementations for the 15 coding questions.

These are interview-sized examples, not production-ready libraries.
Each function states its input contract; tests live in test_algorithm_answers.py.
"""

from __future__ import annotations

import codecs
import ipaddress
import json
import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


def _clone_json(value: Any) -> Any:
    """Clone an acyclic JSON-compatible value without recursive Python calls."""
    if not isinstance(value, (dict, list)):
        return value
    root: Any = {} if isinstance(value, dict) else []
    stack = [(value, root)]
    while stack:
        source, target = stack.pop()
        pairs = source.items() if isinstance(source, dict) else enumerate(source)
        for key, child in pairs:
            cloned = (
                {}
                if isinstance(child, dict)
                else []
                if isinstance(child, list)
                else child
            )
            if isinstance(target, dict):
                target[key] = cloned
            else:
                target.append(cloned)
            if isinstance(child, (dict, list)):
                stack.append((child, cloned))
    return root


def deep_merge(left: dict, right: dict) -> dict:
    """Q121: objects merge, arrays concatenate, other conflicts take right.

    Inputs must be acyclic JSON objects with string keys. Inputs are not mutated.
    Time and space O(N+M) including the independent result.
    """
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise TypeError("Both roots must be JSON objects")
    result = _clone_json(left)
    stack = [(result, right)]
    while stack:
        target, incoming = stack.pop()
        for key, value in incoming.items():
            previous = target.get(key)
            if isinstance(previous, dict) and isinstance(value, dict):
                stack.append((previous, value))
            elif isinstance(previous, list) and isinstance(value, list):
                previous.extend(_clone_json(value))
            else:
                target[key] = _clone_json(value)
    return result


@dataclass
class _CacheNode:
    key: Any = None
    value: Any = None
    previous: _CacheNode | None = None
    next: _CacheNode | None = None


class LRUCache:
    """Q122: explicit hash index and doubly linked list; average O(1) get/put.

    Keys must be hashable. Missing get returns -1. This example is not thread-safe.
    """

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("capacity must be nonnegative")
        self.capacity = capacity
        self._items: dict[Any, _CacheNode] = {}
        self._oldest = _CacheNode()
        self._newest = _CacheNode()
        self._oldest.next = self._newest
        self._newest.previous = self._oldest

    @staticmethod
    def _unlink(node: _CacheNode) -> None:
        node.previous.next = node.next
        node.next.previous = node.previous

    def _append(self, node: _CacheNode) -> None:
        previous = self._newest.previous
        previous.next = node
        node.previous = previous
        node.next = self._newest
        self._newest.previous = node

    def get(self, key: Any) -> Any:
        if key not in self._items:
            return -1
        node = self._items[key]
        self._unlink(node)
        self._append(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if key in self._items:
            node = self._items[key]
            node.value = value
            self._unlink(node)
        else:
            node = _CacheNode(key, value)
            self._items[key] = node
        self._append(node)
        if len(self._items) > self.capacity:
            evicted = self._oldest.next
            self._unlink(evicted)
            del self._items[evicted.key]


def longest_palindrome(text: str) -> str:
    """Q123: center expansion, O(n²) time, O(1) auxiliary space."""
    start, size = 0, 0
    for center in range(len(text)):
        for left, right in ((center, center), (center, center + 1)):
            while left >= 0 and right < len(text) and text[left] == text[right]:
                length = right - left + 1
                if length > size:
                    start, size = left, length
                left -= 1
                right += 1
    return text[start : start + size]


def _reject_constant(value: str):
    raise ValueError(f"Not a standard JSON value: {value}")


class JSONStream:
    """Q124: incremental UTF-8 stream of top-level JSON objects/arrays.

    Scalars are intentionally excluded: an unframed chunk "1" could be the
    prefix of "12". Use NDJSON or length framing when supporting arbitrary JSON.
    Syntax errors may be deferred until finish(); buffers are bounded.
    """

    def __init__(self, max_buffer_bytes: int = 1_048_576):
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self.limit = max_buffer_bytes
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        self._json = json.JSONDecoder(parse_constant=_reject_constant)
        self._text = ""
        self._closed = False

    def _drain(self) -> list:
        results = []
        while True:
            self._text = self._text.lstrip(" \t\r\n")
            if not self._text:
                return results
            if self._text[0] not in "[{":
                raise ValueError("This framing accepts only objects and arrays")
            try:
                value, end = self._json.raw_decode(self._text)
            except json.JSONDecodeError:
                return results
            except RecursionError as exc:
                raise ValueError("JSON nesting exceeds decoder capacity") from exc
            results.append(value)
            self._text = self._text[end:]

    def feed(self, data: bytes) -> list:
        if self._closed:
            raise ValueError("stream is closed")
        if not isinstance(data, bytes):
            raise TypeError("feed expects UTF-8 bytes")
        pending, _ = self._utf8.getstate()
        if len(self._text.encode("utf-8")) + len(pending) + len(data) > self.limit:
            raise ValueError("buffer limit exceeded; use smaller frames/chunks")
        self._text += self._utf8.decode(data, final=False)
        return self._drain()

    def finish(self) -> list:
        if self._closed:
            return []
        self._closed = True
        self._text += self._utf8.decode(b"", final=True)
        result = self._drain()
        if self._text.strip():
            raise ValueError("invalid or incomplete JSON at end of stream")
        return result


def parse_ip(value: str) -> dict:
    """Q125: strict IP literals only, no hostname/CIDR/zone identifier."""
    if not isinstance(value, str):
        raise TypeError("IP address must be a string")
    if "/" in value or "%" in value:
        raise ValueError("CIDR and zone identifiers are outside this contract")
    address = ipaddress.ip_address(value)
    return {"version": address.version, "normalized": str(address)}


@dataclass
class TreeNode:
    value: Any
    left: TreeNode | None = None
    right: TreeNode | None = None


def level_order(root: TreeNode | None) -> list[list]:
    """Q126: O(n) time; queue O(width), returned result O(n)."""
    if root is None:
        return []
    queue = deque([root])
    result = []
    while queue:
        current = []
        for _ in range(len(queue)):
            node = queue.popleft()
            current.append(node.value)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        result.append(current)
    return result


def three_sum(numbers: list[int]) -> list[list[int]]:
    """Q127: unique value triplets; O(n²) time, does not mutate input."""
    values = sorted(numbers)
    result = []
    for index in range(len(values) - 2):
        if index and values[index] == values[index - 1]:
            continue
        if values[index] > 0:
            break
        left, right = index + 1, len(values) - 1
        while left < right:
            total = values[index] + values[left] + values[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([values[index], values[left], values[right]])
                left += 1
                right -= 1
                while left < right and values[left] == values[left - 1]:
                    left += 1
                while left < right and values[right] == values[right + 1]:
                    right -= 1
    return result


@dataclass
class ListNode:
    value: Any
    next: ListNode | None = None


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """Q128: one forward pass, O(n) time/O(1) auxiliary space; invalid n errors."""
    if n <= 0:
        raise ValueError("n must be positive")
    dummy = ListNode(None, head)
    fast = dummy
    for _ in range(n):
        if fast.next is None:
            raise ValueError("n exceeds list length")
        fast = fast.next
    slow = dummy
    while fast.next is not None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next


def max_profit_k(prices: list[int], k: int) -> int:
    """Q129: at most k transactions, no fees/cooldown/short selling.

    O(n*k) time/O(k) space, with O(n) unlimited-transactions fast path.
    """
    if k < 0 or any(price < 0 for price in prices):
        raise ValueError("k and prices must be nonnegative")
    if not prices or k == 0:
        return 0
    if k >= len(prices) // 2:
        return sum(max(0, right - left) for left, right in pairwise(prices))
    cash = [0] * (k + 1)
    hold: list[int | None] = [None] * (k + 1)
    for price in prices:
        old_cash, old_hold = cash[:], hold[:]
        for count in range(1, k + 1):
            bought = old_cash[count - 1] - price
            hold[count] = (
                bought if old_hold[count] is None else max(old_hold[count], bought)
            )
            if old_hold[count] is not None:
                cash[count] = max(old_cash[count], old_hold[count] + price)
    return cash[k]


def _normalize_intervals(intervals: list) -> list[tuple]:
    clean = []
    for left, right in intervals:
        for endpoint in (left, right):
            if (
                not isinstance(endpoint, (int, float))
                or isinstance(endpoint, bool)
                or (isinstance(endpoint, float) and not math.isfinite(endpoint))
            ):
                raise ValueError("endpoints must be finite numbers")
        if left > right:
            raise ValueError("reversed interval")
        if left < right:
            clean.append((left, right))
    result = []
    for left, right in sorted(clean):
        if result and left <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


def interval_difference(first: list, second: list) -> list[tuple]:
    """Q130: set difference of finite half-open intervals [left, right).

    Normalizes overlapping/adjacent intervals. O((n+m)log(n+m)+output) time.
    """
    first, second = _normalize_intervals(first), _normalize_intervals(second)
    result = []
    index = 0
    for left, right in first:
        while index < len(second) and second[index][1] <= left:
            index += 1
        position = left
        scan = index
        while scan < len(second) and second[scan][0] < right:
            cut_left, cut_right = second[scan]
            if position < cut_left:
                result.append((position, min(right, cut_left)))
            position = max(position, cut_right)
            if position >= right:
                break
            scan += 1
        if position < right:
            result.append((position, right))
        index = scan
    return result


def lis_length(numbers: list[int]) -> int:
    """Q131: strictly increasing subsequence length, O(n log n)/O(n)."""
    tails: list[int] = []
    for number in numbers:
        index = bisect_left(tails, number)
        if index == len(tails):
            tails.append(number)
        else:
            tails[index] = number
    return len(tails)


def longest_subarray_sum_k(numbers: list[int], k: int) -> int:
    """Q132: permits negative numbers, average O(n) time/O(n) space."""
    earliest, total, longest = {0: -1}, 0, 0
    for index, value in enumerate(numbers):
        total += value
        if total - k in earliest:
            longest = max(longest, index - earliest[total - k])
        earliest.setdefault(total, index)
    return longest


def reverse_list(head: ListNode | None) -> ListNode | None:
    """Q133: in-place reversal of an acyclic singly linked list."""
    previous = None
    while head is not None:
        after = head.next
        head.next = previous
        previous, head = head, after
    return previous


def longest_valid_parentheses(text: str) -> int:
    """Q134: only '(' and ')' allowed; O(n) time/O(n) auxiliary space."""
    stack = [-1]
    longest = 0
    for index, character in enumerate(text):
        if character == "(":
            stack.append(index)
        elif character == ")":
            stack.pop()
            if not stack:
                stack.append(index)
            else:
                longest = max(longest, index - stack[-1])
        else:
            raise ValueError("only parentheses are accepted")
    return longest


def _num_islands(grid: list[list[str]], breadth_first: bool) -> int:
    if not grid:
        return 0
    rows, columns = len(grid), len(grid[0])
    if any(len(row) != columns for row in grid):
        raise ValueError("grid must be rectangular")
    if any(cell not in {"0", "1"} for row in grid for cell in row):
        raise ValueError("grid cells must be '0' or '1'")
    visited = set()
    count = 0
    for row in range(rows):
        for column in range(columns):
            if grid[row][column] != "1" or (row, column) in visited:
                continue
            count += 1
            frontier = deque([(row, column)])
            visited.add((row, column))
            while frontier:
                x, y = frontier.popleft() if breadth_first else frontier.pop()
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if (
                        0 <= nx < rows
                        and 0 <= ny < columns
                        and grid[nx][ny] == "1"
                        and (nx, ny) not in visited
                    ):
                        visited.add((nx, ny))
                        frontier.append((nx, ny))
    return count


def num_islands_dfs(grid: list[list[str]]) -> int:
    """Q135: explicit-stack DFS; four-neighbor, input preserved."""
    return _num_islands(grid, breadth_first=False)


def num_islands_bfs(grid: list[list[str]]) -> int:
    """Q135: queue BFS; four-neighbor, input preserved."""
    return _num_islands(grid, breadth_first=True)
