"""Deterministic unit and randomized-oracle checks, without network access."""

import itertools
import json
import random
from copy import deepcopy
from functools import lru_cache

import pytest
from algorithm_answers import (
    JSONStream,
    ListNode,
    LRUCache,
    TreeNode,
    deep_merge,
    interval_difference,
    level_order,
    lis_length,
    longest_palindrome,
    longest_subarray_sum_k,
    longest_valid_parentheses,
    max_profit_k,
    num_islands_bfs,
    num_islands_dfs,
    parse_ip,
    remove_nth_from_end,
    reverse_list,
    three_sum,
)


def linked(values):
    result = None
    for value in reversed(values):
        result = ListNode(value, result)
    return result


def values(head):
    result = []
    while head is not None:
        result.append(head.value)
        head = head.next
    return result


def test_deep_merge_rules_and_no_mutation():
    left = {"a": {"x": 1}, "b": [{"v": 1}], "c": 1, "d": None}
    right = {"a": {"y": 2}, "b": [2], "c": {"new": 3}, "d": False}
    saved_left, saved_right = deepcopy(left), deepcopy(right)
    result = deep_merge(left, right)
    assert result == {
        "a": {"x": 1, "y": 2},
        "b": [{"v": 1}, 2],
        "c": {"new": 3},
        "d": False,
    }
    result["b"][0]["v"] = 100
    result["c"]["new"] = 100
    assert left == saved_left and right == saved_right
    with pytest.raises(TypeError):
        deep_merge([], {})


def test_merge_handles_nesting_beyond_python_recursion_limit():
    left, right = {}, {}
    a, b = left, right
    for _ in range(2500):
        a["n"], b["n"] = {}, {}
        a, b = a["n"], b["n"]
    a["left"], b["right"] = 1, 2
    result = deep_merge(left, right)
    for _ in range(2500):
        result = result["n"]
    assert result == {"left": 1, "right": 2}


def test_lru_eviction_and_update():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert cache.get("b") == -1
    cache.put("a", 9)
    cache.put("d", 4)
    assert cache.get("c") == -1
    assert cache.get("a") == 9
    empty = LRUCache(0)
    empty.put(1, 1)
    assert empty.get(1) == -1
    with pytest.raises(ValueError):
        LRUCache(-1)


@pytest.mark.parametrize("text", ["", "x", "abba", "babad", "aaaa", "你好好你", "abc"])
def test_palindrome_against_brute_force(text):
    result = longest_palindrome(text)
    expected = max(
        (
            len(text[a:b])
            for a in range(len(text) + 1)
            for b in range(a, len(text) + 1)
            if text[a:b] == text[a:b][::-1]
        ),
        default=0,
    )
    assert result == result[::-1] and len(result) == expected and result in text


def test_json_stream_all_byte_boundaries_and_bytewise():
    expected = [{"中文": 'x\\"y', "a": [1, True, None]}, [2, {"b": "}"}], {}]
    payload = (
        " \n" + "".join(json.dumps(x, ensure_ascii=False) for x in expected)
    ).encode()
    for split in range(len(payload) + 1):
        stream = JSONStream()
        got = stream.feed(payload[:split]) + stream.feed(payload[split:])
        assert got + stream.finish() == expected
    stream = JSONStream()
    result = []
    for byte in payload:
        result.extend(stream.feed(bytes([byte])))
    assert result + stream.finish() == expected
    assert stream.finish() == []
    with pytest.raises(ValueError):
        stream.feed(b"{}")


@pytest.mark.parametrize(
    "payload", [b'{"x":', b'{"x":]}', b"[1,]", b'{"x":"unterminated']
)
def test_json_stream_incomplete_or_invalid(payload):
    stream = JSONStream()
    stream.feed(payload)
    with pytest.raises(ValueError):
        stream.finish()


@pytest.mark.parametrize("payload", [b"12", b"true", b'"hello"', b"null", b'{"x":NaN}'])
def test_json_stream_rejects_unframed_scalars_and_nonstandard_constants(payload):
    with pytest.raises(ValueError):
        JSONStream().feed(payload)


def test_json_stream_bounds_and_utf8():
    with pytest.raises(ValueError):
        JSONStream(3).feed(b'{"long": 1}')
    with pytest.raises(ValueError):
        JSONStream(0)
    stream = JSONStream()
    stream.feed(b'{"x":"\xe4')
    with pytest.raises(UnicodeDecodeError):
        stream.finish()
    with pytest.raises(TypeError):
        JSONStream().feed("{}")


@pytest.mark.parametrize(
    "literal,version", [("127.0.0.1", 4), ("::1", 6), ("2001:db8::1", 6)]
)
def test_ip(literal, version):
    assert parse_ip(literal)["version"] == version


@pytest.mark.parametrize(
    "literal",
    [
        "256.1.1.1",
        "01.2.3.4",
        "localhost",
        "1.2.3",
        "::1/128",
        "fe80::1%en0",
        " 1.2.3.4",
    ],
)
def test_ip_rejects_invalid_and_out_of_contract(literal):
    with pytest.raises(ValueError):
        parse_ip(literal)


def test_tree_levels():
    tree = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
    assert level_order(tree) == [[1], [2, 3], [4]]
    assert level_order(None) == []


def test_three_sum_random_oracle():
    rng = random.Random(156)
    for _ in range(100):
        numbers = [rng.randrange(-4, 5) for _ in range(rng.randrange(9))]
        before = numbers[:]
        expected = {
            tuple(sorted(combination))
            for combination in itertools.combinations(numbers, 3)
            if sum(combination) == 0
        }
        got = three_sum(numbers)
        assert {tuple(x) for x in got} == expected
        assert len(got) == len(expected)
        assert numbers == before


def test_list_removal_and_reversal():
    for size in range(1, 9):
        for n in range(1, size + 1):
            original = list(range(size))
            assert values(remove_nth_from_end(linked(original), n)) == (
                original[: size - n] + original[size - n + 1 :]
            )
        assert values(reverse_list(linked(list(range(size))))) == list(
            reversed(range(size))
        )
    assert reverse_list(None) is None
    for head, n in [(None, 1), (linked([1]), 0), (linked([1]), 2)]:
        with pytest.raises(ValueError):
            remove_nth_from_end(head, n)


def test_stock_against_exhaustive_state_search():
    rng = random.Random(129)
    for _ in range(150):
        prices = [rng.randrange(10) for _ in range(rng.randrange(9))]
        k = rng.randrange(5)

        @lru_cache(None)
        def brute(day, remaining, holding, fixed_prices=tuple(prices)):
            if day == len(fixed_prices):
                return -(10**9) if holding else 0
            best = brute(day + 1, remaining, holding)
            if holding and remaining:
                best = max(
                    best, fixed_prices[day] + brute(day + 1, remaining - 1, False)
                )
            elif not holding and remaining:
                best = max(best, -fixed_prices[day] + brute(day + 1, remaining, True))
            return best

        assert max_profit_k(prices, k) == brute(0, k, False)
    with pytest.raises(ValueError):
        max_profit_k([1], -1)
    large = 10**400
    assert max_profit_k([0, large, 0, 0, large, 0], 2) == 2 * large


def test_interval_difference_random_set_oracle():
    rng = random.Random(130)
    for _ in range(200):
        first, second = [], []
        for intervals in (first, second):
            for _ in range(rng.randrange(7)):
                intervals.append(tuple(sorted((rng.randrange(10), rng.randrange(10)))))
        result = interval_difference(first, second)
        expected = {x for a, b in first for x in range(a, b)} - {
            x for a, b in second for x in range(a, b)
        }
        assert {x for a, b in result for x in range(a, b)} == expected
        assert all(a < b for a, b in result)
        assert all(result[i][1] < result[i + 1][0] for i in range(len(result) - 1))
    assert interval_difference([(0.0, 2.0)], [(0.5, 1.5)]) == [(0.0, 0.5), (1.5, 2.0)]
    for invalid in [[(2, 1)], [(0, float("nan"))], [(False, 1)]]:
        with pytest.raises(ValueError):
            interval_difference(invalid, [])


def test_lis_against_subsequence_enumeration():
    rng = random.Random(131)
    for _ in range(100):
        numbers = [rng.randrange(6) for _ in range(rng.randrange(9))]
        expected = 0
        for mask in range(1 << len(numbers)):
            sequence = [x for i, x in enumerate(numbers) if mask & (1 << i)]
            if all(a < b for a, b in itertools.pairwise(sequence)):
                expected = max(expected, len(sequence))
        assert lis_length(numbers) == expected


def test_subarray_against_brute_force():
    rng = random.Random(132)
    for _ in range(200):
        numbers = [rng.randrange(-4, 5) for _ in range(rng.randrange(10))]
        k = rng.randrange(-5, 6)
        expected = max(
            (
                b - a
                for a in range(len(numbers))
                for b in range(a + 1, len(numbers) + 1)
                if sum(numbers[a:b]) == k
            ),
            default=0,
        )
        assert longest_subarray_sum_k(numbers, k) == expected


def test_parentheses_against_brute_force():
    for size in range(8):
        for characters in itertools.product("()", repeat=size):
            text = "".join(characters)
            expected = 0
            for left in range(size):
                balance = 0
                for right in range(left, size):
                    balance += 1 if text[right] == "(" else -1
                    if balance < 0:
                        break
                    if balance == 0:
                        expected = max(expected, right - left + 1)
            assert longest_valid_parentheses(text) == expected
    with pytest.raises(ValueError):
        longest_valid_parentheses("x")


def test_islands_against_union_find():
    rng = random.Random(135)
    for _ in range(100):
        rows, columns = rng.randrange(6), rng.randrange(6)
        grid = [[rng.choice("01") for _ in range(columns)] for _ in range(rows)]
        original = deepcopy(grid)
        parent = {
            (r, c): (r, c)
            for r in range(rows)
            for c in range(columns)
            if grid[r][c] == "1"
        }

        def root(cell, links=parent):
            while links[cell] != cell:
                cell = links[cell]
            return cell

        for r, c in parent:
            for neighbor in ((r + 1, c), (r, c + 1)):
                if neighbor in parent:
                    parent[root((r, c))] = root(neighbor)
        expected = len({root(cell) for cell in parent})
        assert num_islands_dfs(grid) == expected
        assert num_islands_bfs(grid) == expected
        assert grid == original
    for invalid in [[["1"], []], [["x"]]]:
        with pytest.raises(ValueError):
            num_islands_bfs(invalid)
