"""
Implement a simple consistent hash, mapping an example set of objects
to a set of storage devices.  Each storage device is represented
by several notional "chunks" each with their own hash code.

Using a representative set of object sizes, demonstrate the device
usage imbalance that can occur.
"""

import bisect
import copy
import datetime
import hashlib
import numpy as np
import os
import unittest

from collections import namedtuple, defaultdict
from functools import reduce

try:
    import cPickle as pickle
except ImportError:
    import pickle


class DeviceMapper(object):
    """
    Represent a chunk mapping, mapping keys to a device.
    """

    HashSize = namedtuple("HashSize", ["hash", "size"])
    FreeSizeDeviceId = namedtuple("FreeSizeDeviceId", ["free_size", "device_id"])

    def __init__(self):
        # We separate lists of hash values for chunks and devices they map to, so we can easily bisect
        self._device_count = 0

        # List of chunk hashes, formed from the hash of 'osdN:C'
        self._chunk_map = []

        # Parallel list, with the device number for each hash
        self._device_map = []

    def create(self, chunks, device_count):
        """
        Create a fresh device mapping.
        :param chunks: The number of chunks to include.
        :param device_count: The number of devices to include.
        """
        self._device_count = device_count
        self._chunk_map = []
        self._device_map = []

        # Quicker to sort once, rather than insert into sorted order
        mapping = []
        for device in range(device_count):
            for chunk in range(chunks):
                name = 'osd{}:{}'.format(device, chunk)
                hash_value = md5_hash(name)
                mapping.append((hash_value, device))
        mapping.sort()

        for hash_value, device in mapping:
            self._chunk_map.append(hash_value)
            self._device_map.append(device)

    def get_state(self):
        """
        Get the mapping state to send to other devices on the theoretical network.
        :return: tuple(device_count, chunk_map, device_map)
        """
        return self._device_count, self._chunk_map, self._device_map

    def set_state(self, state):
        """
        Set the chunk and device mapping state.
        :param state: tuple(device_count, chunk_map, device_map)
        """
        self._device_count, self._chunk_map, self._device_map = state

    def locate_chunk(self, hash_value):
        """
        Locate the chunk a file is on.
        :param hash_value: The hashed file name.
        :return: The chunk ID.
        """
        chunk_id = bisect.bisect(self._chunk_map, hash_value)
        if chunk_id == len(self._device_map):
            chunk_id = 0
        return chunk_id

    def locate_device(self, hash_value):
        """
        Locate the device a file is on.
        :param hash_value: The hashed file name.
        :return: The device ID.
        """
        device_id = self._device_map[self.locate_chunk(hash_value)]
        return device_id

    def moving(self, l, r, need_free_space, sorted_file_data, map_device_id_to_half_intervals, new_chunks):
        """
        Determine which files to move
        :param l: int Index of device in need_free_space list where we move files
        :param r: int Index of device in need_free_space list we are moving files from
        :param need_free_space: list[(int, int)] fisrt - need_free, second - device_id
        :param map_device_id_to_half_intervals: dict(int, list(int, int))
               map between device id and its list of half intervals of chunks
        :param new_chunks: dict(int, int) key = new chunk, value = device id
        :return: (int, int) first - 0 or 1, 0 - need move left pointer, 1 - move right pointer.
                 second - amount of bytes moved at this funcction
        """
        # return 0 -left, 1 - right
        n_files = len(sorted_file_data)
        total_bytes_moved = 0
        can_free = -need_free_space[l][0]
        need_free = need_free_space[r][0]
        device_id_l = need_free_space[l][1]
        device_id_r = need_free_space[r][1]
        device_half_intervals = map_device_id_to_half_intervals[device_id_r]
        # loop through all half intervals of chunks of right device
        while device_half_intervals:
            f_chunk, s_chunk = device_half_intervals[-1]
            f_file_idx = bisect.bisect_left(sorted_file_data, (f_chunk, 0))
            s_file_idx = bisect.bisect_left(sorted_file_data, (s_chunk, 0))
            cur_file_idx = f_file_idx
            prev_hash = sorted_file_data[f_file_idx][0] + 1
            # loop through files which are contained at this half interval
            while (cur_file_idx % n_files) < s_file_idx:
                file_hash, file_size = sorted_file_data[cur_file_idx]
                if file_size > can_free:
                    # if we can not move this file to left device,
                    # because left device do not have enough free space
                    # we return that we need to move left pointer
                    # and use next device where to move files
                    need_free_space[l] = (-can_free, device_id_l)
                    need_free_space[r] = (need_free, device_id_r)
                    device_half_intervals.pop()
                    if cur_file_idx != f_file_idx:
                        new_chunks[prev_hash] = device_id_l
                        device_half_intervals.append((file_hash, s_chunk))
                    return 0, total_bytes_moved
                if need_free <= 0:
                    # if used size of right device already less then threshold
                    # we return that we need move right pointer
                    # and move files from next device
                    need_free_space[l] = (-can_free, device_id_l)
                    need_free_space[r] = (need_free, device_id_r)
                    device_half_intervals.pop()
                    if cur_file_idx != f_file_idx:
                        new_chunks[prev_hash] = device_id_l
                    return 1, total_bytes_moved
                can_free -= file_size
                need_free -= file_size
                total_bytes_moved += file_size
                cur_file_idx += 1
                prev_hash = file_hash + 1
            need_free_space[l] = (-can_free, device_id_l)
            need_free_space[r] = (need_free, device_id_r)
            device_half_intervals.pop()
            if cur_file_idx != f_file_idx:
                new_chunks[s_chunk] = device_id_l
        return 1, total_bytes_moved

    def rebalance(self, file_data, device_size, max_load_factor):
        """
        Rebalance the data on each device to bring each device's total load under the max_load_factor.
        :param file_data: list(tuple(file hash, file size)) mappings.
        :param device_size: int The size of each device.
        :param max_load_factor: The maximum load per device in range [0.0 no data, 1.0 full disk]
        :return: The total bytes moved.
        """
        used_device_capacity = {device_id: 0 for device_id in self._device_map}
        sorted_file_data = sorted(file_data)
        for file_hash, file_size in file_data:
            device_id = self.locate_device(file_hash)
            used_device_capacity[device_id] += file_size

        device_to_chunk = defaultdict(list)
        for i, device_id in enumerate(self._device_map):
            chunk = self._chunk_map[i]
            device_to_chunk[device_id].append(chunk)

        # calculate how many space we need to free in each device
        # if need_free < 0 it means that this device have this amount of free space until threshold
        need_free_space = []
        need_size = max_load_factor * device_size
        for device_id, used_capacity in used_device_capacity.items():
            need_free = used_capacity - need_size
            need_free_space.append((need_free, device_id))
        need_free_space.sort()

        # build half intervals of chunks for each device
        # second_chunk - chunk of this devise
        # first_chunk - previous chunk in sorted order of all chunks
        map_device_id_to_half_intervals = {}
        for device_id, chunks in device_to_chunk.items():
            half_intervals = []
            for chunk in chunks:
                second_idx = bisect.bisect_left(self._chunk_map, chunk)
                second_chunk = self._chunk_map[second_idx]
                first_chunk = self._chunk_map[(second_idx - 1 + len(self._chunk_map)) % len(self._chunk_map)]
                half_intervals.append((first_chunk, second_chunk))
            map_device_id_to_half_intervals[device_id] = half_intervals

        # move two pointers on the sorted list
        # r - right pointer means the device we are moving files from
        # l - left pointer means the device where we move files
        # thus we copy files from a device with a higher fullness to a device with a lower fullness
        new_chunks = {}
        total_bytes_moved = 0
        l = 0
        r = len(need_free_space) - 1
        while l < r and need_free_space[r][0] > 0:
            move, moved_bytes = self.moving(
                l, r, need_free_space, sorted_file_data, map_device_id_to_half_intervals, new_chunks
            )
            if move == 0:
                l += 1
            elif move == 1:
                r -= 1
            else:
                raise Exception("self.moving() returned unexpected value")
            total_bytes_moved += moved_bytes

        # set new states
        # create new lists to do not insert each time to existing
        mapping = []
        for chunk, device_id in new_chunks.items():
            mapping.append((chunk, device_id))
        for i, chunk in enumerate(self._chunk_map):
            device_id = self._device_map[i]
            if chunk not in new_chunks:
                mapping.append((chunk, device_id))
        mapping.sort()

        self._chunk_map = []
        self._device_map = []
        for hash_value, device in mapping:
            self._chunk_map.append(hash_value)
            self._device_map.append(device)
        return total_bytes_moved


class DeviceMapperTests(unittest.TestCase):
    """
    Methods to generate test data and test the DeviceMapper.rebalance implementation.
    """

    _seed = 0
    _device_count = 20
    _file_count = 1000000
    _device_size = int(2e12)
    _chunk_size = int(10e9)
    _max_factor_load = 0.9
    _max_mapping_expansion = 2.0
    _cache_file = 'data-{seed}.pkl'

    def test_device_mapper(self):
        """
        A basic test for the device mapper.
        :raises AssertionError: On test fail.
        """
        file_data, mapper = self._get_data(self._seed)
        old_file_data = copy.deepcopy(file_data)
        old_mapper = copy.deepcopy(mapper)

        total_device_size = self._device_count * self._device_size
        total_file_size = float(np.sum([file_size for (hash_value, file_size) in file_data]))

        print('Files: {:.2f}T'.format(float(total_file_size) / (1 << 40)))
        print('Devices: {:.2f}T'.format(total_device_size / (1 << 40)))
        print('Use: {:.3f}%'.format(100.0 * total_file_size / total_device_size))

        print('Device usage before rebalance:')
        self._print_device_usage(file_data, mapper)

        start = datetime.datetime.now()
        bytes_moved = mapper.rebalance(file_data, self._device_size, self._max_factor_load)
        end = datetime.datetime.now()
        print('Total data movement: {:.2f}G'.format(bytes_moved / float(1 << 30)))
        print('Time taken: {}'.format(end - start))

        print('Device usage after rebalance:')
        self._print_device_usage(file_data, mapper)
        self._post_rebalance_checks(old_file_data, old_mapper, file_data, mapper, bytes_moved)

    def _post_rebalance_checks(self, old_file_data, old_mapper, file_data, mapper, bytes_moved):
        """
        Check the resulting rebalance is valid.
        :param old_file_data: The old file data.
        :param old_mapper: The old device mapper.
        :param file_data: The current file data.
        :param mapper: The current device mapper.
        :param bytes_moved: The number of bytes moved.
        :raises AssertionError: On test fail.
        """
        print('Running tests...')

        # 0. Check that file_data is unchanged.
        self.assertEqual(sorted(old_file_data), sorted(file_data), 'File data should not have changed.')
        self.assertNotEqual(old_mapper.get_state(), mapper.get_state(), 'Mappings are unchanged?')

        # Compute device usage before and after rebalance.
        old_device_usages = np.zeros(self._device_count, dtype=np.int64)
        device_usages = np.zeros(self._device_count, dtype=np.int64)
        for hash_value, file_size in file_data:
            old_device_usages[old_mapper.locate_device(hash_value)] += file_size
            device_usages[mapper.locate_device(hash_value)] += file_size

        # 1. Check we haven't lost data.
        self.assertEqual(sum(old_device_usages), sum(device_usages), 'Files/chunks have been lost!')

        # 2. Check if the hash ring and chunk-to-device mapping are valid.
        self.assertEqual(mapper._chunk_map, sorted(mapper._chunk_map), 'Chunk map not valid.')
        self.assertEqual(len(mapper._chunk_map), len(mapper._device_map), 'Device map not valid.')

        # 3. Check that no new devices have been added.
        self.assertTrue(
            0 <= min(mapper._device_map) <= max(mapper._device_map) < self._device_count,
            'New devices were added.')

        # 4. Check the hash ring mapping size hasn't expanded rapidly.
        initial_chunks = self._device_count * (self._device_size // self._chunk_size)
        max_chunk_map_size = initial_chunks * self._max_mapping_expansion
        self.assertLessEqual(
            len(mapper._chunk_map), max_chunk_map_size,
            'Chunk mapping size has expanded by over {}x'.format(self._max_mapping_expansion))

        # 5. Check that the devices are balanced now.
        self.assertTrue(
            all([du <= self._device_size * self._max_factor_load for du in device_usages]),
            'Not all devices are balanced under the max factor load.')

        # 6. Check the amount of data moved.
        real_bytes_moved = sum(
            file_size for hash_value, file_size in file_data
            if old_mapper.locate_device(hash_value) != mapper.locate_device(hash_value))

        theoretical_min = self._theoretical_min_movement(old_file_data, old_mapper)
        limit = 2 * theoretical_min
        print('Data movement theoretical min: {:.2f}G, actual: {:.2f}G, max: {:.2f}G'.format(
            theoretical_min / float(1 << 30),
            real_bytes_moved / float(1 << 30),
            limit / float(1 << 30)))
        self.assertLessEqual(real_bytes_moved, limit, 'Bytes moved over theoretical limit.')
        self.assertLessEqual(real_bytes_moved, bytes_moved, 'DeviceMapper underestimating the bytes moved.')

        print('Tests passed.')

    def _theoretical_min_movement(self, file_data, mapper):
        """
        Calculate the theoretical minimum data movement.
        :param file_data: The file data.
        :param mapper: The device mapper.
        :return: The min theoretical movement between devices.
        """
        used_device_capacity = defaultdict(int)
        for hash_value, file_size in file_data:
            used_device_capacity[mapper.locate_device(hash_value)] += file_size
        need_size = self._device_size * self._max_factor_load
        theoretical_min = reduce((lambda a, x: a + max(x - need_size, 0)), used_device_capacity.values(), 0)
        return theoretical_min

    def _get_data(self, seed):
        """
        Get the dataset for a particular seed.
        :param seed: The seed the dataset is generated with.
        :return: tuple(file_data, mapper).
        """
        cache_file = self._cache_file.format(seed=seed)
        if not os.path.exists(cache_file):
            print('State file {} doesn\'t exist, creating and saving an initial data set.'.format(cache_file))
            file_data, mapper = self._create_data(seed)
            self._save_data(cache_file, file_data, mapper)
        else:
            print('Loading data set {}'.format(cache_file))
            file_data, mapper = self._load_data(cache_file)
        return file_data, mapper

    def _create_data(self, seed):
        """
        Create a representative random dataset and mapping.
        :param seed: An integer PRNG seed.
        :return: tuple(list(tuple(file_hash, file_size)), DeviceMapper)
        """
        # Repeatable
        np.random.seed(seed)

        # Create a normally distributed set of files sizes, clipped to 1B -> 100GB
        file_sizes = np.clip(np.exp(3.3 * np.random.randn(self._file_count) + 12), 1, 100 * 1 << 30).astype(np.int64)

        # Calculate file name hashes based on fictional names
        file_name_hashes = [md5_hash('file-{}'.format(file_number)) for file_number in range(self._file_count)]

        mapper = DeviceMapper()
        mapper.create(self._device_size // self._chunk_size, self._device_count)

        return list(zip(file_name_hashes, file_sizes)), mapper

    def _save_data(self, file_name, file_data, mapper):
        """
        Save the random generated dataset to disk to avoid generating it every time.
        :param file_name: The name of the data file (data.pkl).
        :param file_data: The file mappings.
        :param mapper: The mapper state to save.
        """
        with open(file_name, 'wb') as file:
            pickle.dump((file_data, mapper.get_state()), file, pickle.HIGHEST_PROTOCOL)

    def _load_data(self, file_name):
        """
        Load the dataset cache from disk.
        :param file_name: The name of the data file (data.pkl)
        :return: The file data and initial mapper state.
        """
        with open(file_name, 'rb') as file:
            file_data, mapper_state = pickle.load(file)
        mapper = DeviceMapper()
        mapper.set_state(mapper_state)
        return file_data, mapper

    def _print_device_usage(self, file_data, mapper):
        """
        Print the device usage to console.
        :param mapper: The device mapper.
        :param file_data: The file data.
        """
        used = np.zeros(self._device_count, dtype=np.int64)

        for hash_value, file_size in file_data:
            used[mapper.locate_device(hash_value)] += file_size

        for device in range(self._device_count):
            print('{:2d} {:7.2f}G {}: {:4.1f}%'.format(
                device,
                float(used[device]) / (1 << 30),
                used[device],
                100.0 * float(used[device]) / self._device_size))


def md5_hash(string):
    """
    Compute the 128-bit MD5 hash for a string.
    :param string: The string to hash.
    :return: The int value of the hash.
    """
    return int(hashlib.md5(string.encode()).hexdigest(), 16)


def main():
    """
    Run the unittests for DeviceMapper.
    """
    tests = DeviceMapperTests('test_device_mapper')
    tests.test_device_mapper()


if __name__ == '__main__':
    main()
