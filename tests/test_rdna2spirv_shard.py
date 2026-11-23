"""rdna2spirv.shard_bucket: the buckets partition the listing, and they are balanced by size.

    python -I tests/test_rdna2spirv_shard.py

--shard used to take `files[index::count]`. Shaders differ in size by two orders of magnitude, so one shard could
collect most of the work and the other processes waited for it; the partition below is what the parallel stage
actually uses, so it is checked here without translating anything (only file sizes matter).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import rdna2spirv  # noqa: E402


class ShardBucketTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, sizes):
        """Files of the given sizes, named in an order unrelated to their size."""
        paths = []
        for index, size in enumerate(sizes):
            path = os.path.join(self.root, "ps_s%03d_%d.s" % (len(sizes) - index, size))
            with open(path, "wb") as f:
                f.write(b"\0" * size)
            paths.append(path)
        return sorted(paths)

    def test_every_file_lands_in_exactly_one_bucket(self):
        files = self.make([5, 40, 7, 900, 12, 3, 60, 1])
        for count in (1, 2, 3, 4, 8, 16):
            buckets = [rdna2spirv.shard_bucket(files, index, count) for index in range(count)]
            flat = sorted(path for bucket in buckets for path in bucket)
            self.assertEqual(flat, files, count)
            self.assertEqual(len(set(flat)), len(flat), count)
            for bucket in buckets:
                self.assertEqual(bucket, sorted(bucket), "a bucket is still handed out in name order")

    def test_the_partition_is_the_same_in_every_process(self):
        files = self.make([5, 40, 7, 900, 12, 3, 60, 1])
        first = [rdna2spirv.shard_bucket(files, index, 4) for index in range(4)]
        second = [rdna2spirv.shard_bucket(list(files), index, 4) for index in range(4)]
        self.assertEqual(first, second)

    def test_a_skewed_listing_is_balanced_better_than_round_robin(self):
        # One shader that is 60% of the bytes and a long tail of small ones: the case that costs a build wall time.
        files = self.make([6000] + [10] * 20 + [100] * 4)
        size = {path: os.path.getsize(path) for path in files}
        total = sum(size.values())
        for count in (2, 4, 8):
            buckets = [rdna2spirv.shard_bucket(files, index, count) for index in range(count)]
            span = max(sum(size[path] for path in bucket) for bucket in buckets)
            round_robin = max(sum(size[path] for path in files[index::count]) for index in range(count))
            # The heaviest bucket carries the largest file plus one share of the rest, never a second large share.
            self.assertLessEqual(span, total / count + max(size.values()), count)
            self.assertLess(span, round_robin, "no better than files[::%d]" % count)

    def test_an_unreadable_size_does_not_drop_the_file(self):
        files = self.make([5, 40, 7])
        gone = os.path.join(self.root, "ps_gone_0.s")
        files = sorted(files + [gone])          # never created: getsize fails, the path still has to be scheduled
        buckets = [rdna2spirv.shard_bucket(files, index, 2) for index in range(2)]
        flat = sorted(path for bucket in buckets for path in bucket)
        self.assertEqual(flat, files)

    def test_one_shard_takes_everything(self):
        files = self.make([5, 40, 7])
        self.assertEqual(rdna2spirv.shard_bucket(files, 0, 1), files)


if __name__ == "__main__":
    unittest.main()
