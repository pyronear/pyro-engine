# Copyright (C) 2022-2026, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
"""NCNN scratch storage that merges adjacent freed regions instead of caching each size."""

import bisect
import ctypes

import ncnn

_address = ctypes.pythonapi.PyCapsule_GetPointer
_address.argtypes = [ctypes.py_object, ctypes.c_char_p]
_address.restype = ctypes.c_void_p
_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_capsule.restype = ctypes.py_object


class ArenaAllocator(ncnn.PoolAllocator):
    """Use one aligned arena, with ordinary NCNN allocations when it cannot fit a request.

    NCNN's Python allocator API exchanges native pointers as capsules. The backing
    allocation belongs to NCNN; subregions keep its 64-byte alignment and overread guard.
    The owner must keep this allocator alive until every extractor and tensor is released.
    """

    def __init__(self, size_mb: int) -> None:
        super().__init__()
        if isinstance(size_mb, bool) or not isinstance(size_mb, int) or size_mb <= 0:
            raise ValueError("ncnn_memory_mb must be a positive integer")
        self._native = ncnn.PoolAllocator()
        self._size = size_mb * 1024**2
        self._backing = self._native.fastMalloc(self._size + 63)
        if self._backing is None:
            raise MemoryError("Unable to allocate NCNN scratch arena")
        self._base = (int(_address(self._backing, None)) + 63) // 64 * 64
        self._free = [(0, self._size)]
        self._live: dict[int, tuple[int, object, int | None]] = {}

    def fastMalloc(self, size: int) -> object:
        # NCNN kernels may read 64 bytes beyond a tensor. Keep that space separate
        # from the next tensor, then align every region to the strictest NCNN boundary.
        capacity = (size + 127) // 64 * 64
        choices = [(length, i, offset) for i, (offset, length) in enumerate(self._free) if length >= capacity]
        offset: int | None
        if choices:
            length, index, offset = min(choices)
            self._free.pop(index)
            if length > capacity:
                self._free.insert(index, (offset + capacity, length - capacity))
            pointer = _capsule(self._base + offset, None, None)
        else:
            pointer = self._native.fastMalloc(size)
            offset = None
        if pointer is not None:
            self._live[int(_address(pointer, None))] = (capacity, pointer, offset)
        return pointer

    def fastFree(self, pointer: object) -> None:
        if pointer is None:
            return
        capacity, original, offset = self._live.pop(int(_address(pointer, None)))
        if offset is None:
            self._native.fastFree(original)
            self._native.clear()
            return
        index = bisect.bisect_left(self._free, (offset, 0))
        if index and sum(self._free[index - 1]) == offset:
            offset, length = self._free.pop(index - 1)
            capacity += length
            index -= 1
        if index < len(self._free) and offset + capacity == self._free[index][0]:
            _, length = self._free.pop(index)
            capacity += length
        self._free.insert(index, (offset, capacity))

    def __del__(self) -> None:
        if getattr(self, "_backing", None) is not None:
            self._native.fastFree(self._backing)
            self._native.clear()
