from __future__ import annotations

import pytest

from gala_sim.timing.modules import CacheBackpressure, CacheLookup, SemanticCacheState


def test_semantic_cache_tracks_miss_merge_multicast_and_release() -> None:
    cache = SemanticCacheState.create(capacity=2, directory_banks=2, sector_bytes=64)
    key = (9, 3)
    assert cache.request(key, remaining_uses=2) is CacheLookup.MISS
    assert cache.request(key, remaining_uses=2) is CacheLookup.MERGED
    cache.fill_complete(key, remaining_uses=2)
    cache.begin_multicast(key, destinations=2)
    cache.complete_read(key)
    cache.complete_read(key)
    cache.complete_read(key)
    assert cache.close(key) is False
    cache.complete_read(key)
    assert cache.close(key) is True
    assert cache.counters["miss_merges"] == 1
    assert cache.counters["fills"] == 1
    assert cache.counters["releases"] == 1


def test_semantic_cache_refuses_capacity_overflow() -> None:
    cache = SemanticCacheState.create(capacity=1, directory_banks=1, sector_bytes=64)
    cache.request((1, 0), remaining_uses=1)
    with pytest.raises(CacheBackpressure):
        cache.request((2, 0), remaining_uses=1)


def test_semantic_cache_consumes_exact_workset_remaining_uses() -> None:
    cache = SemanticCacheState.create(capacity=1, directory_banks=1, sector_bytes=64)
    key = (3, 0)
    assert cache.request(
        key, remaining_uses=2, workset_total_uses=2,
    ) is CacheLookup.MISS
    cache.fill_complete(key, remaining_uses=2, active_reads=1)
    cache.complete_read(key)
    assert cache.request(
        key, remaining_uses=1, workset_total_uses=2,
    ) is CacheLookup.HIT
    cache.complete_read(key)
    assert cache.close(key) is True
