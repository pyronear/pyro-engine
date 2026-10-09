# Copyright (C) 2022-2026, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
"""Guard native memory boundaries, overflow, and borrowed tensor lifetimes."""

import ctypes

import ncnn
import numpy as np
import pytest
from pyro_predictor._allocator import ArenaAllocator, _address


@pytest.mark.parametrize("size", [0, -1, False, 1.5])
def test_arena_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="positive integer"):
        ArenaAllocator(size)


def test_arena_reuses_adjacent_regions_without_touching_live_data():
    pool = ArenaAllocator(1)
    rng = np.random.default_rng(42)
    live = []
    for _ in range(200):
        if live and rng.random() < 0.5:
            index = int(rng.integers(len(live)))
            pointer, size, value = live.pop(index)
            assert ctypes.string_at(_address(pointer, None), size + 64) == bytes([value]) * (size + 64)
            pool.fastFree(pointer)
        else:
            size = int(rng.choice([1, 63, 1024, 500_000, 2_000_000]))
            pointer = pool.fastMalloc(size)
            address = _address(pointer, None)
            assert address % 64 == 0 or pool._live[address][2] is None
            for other, other_size, _ in live:
                start = _address(other, None)
                assert address + size + 64 <= start or start + other_size + 64 <= address
            value = int(rng.integers(256))
            ctypes.memset(address, value, size + 64)
            live.append((pointer, size, value))
    for pointer, size, value in live:
        assert ctypes.string_at(_address(pointer, None), size + 64) == bytes([value]) * (size + 64)
        pool.fastFree(pointer)
    assert pool._live == {}
    assert pool._free == [(0, pool._size)]


@pytest.mark.parametrize("width", [256, 2048])
def test_arena_and_fallback_outlive_a_borrowed_numpy_view(width):
    pool = ArenaAllocator(1)
    tensor = ncnn.Mat(width, 128, 3, allocator=pool)
    view = np.asarray(tensor)
    view[:] = 0.125
    assert len(pool._live) == 1
    assert (next(iter(pool._live.values()))[2] is None) == (width == 2048)
    del tensor
    assert len(pool._live) == 1
    assert np.all(view == 0.125)
    del view
    assert pool._live == {}
