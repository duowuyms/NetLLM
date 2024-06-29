import numpy as np
from baseline_special.utils.constants import (TOTAL_VIDEO_CHUNK, VIDEO_CHUNK_LEN)

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 998244353
BITRATE_LEVELS = 6
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None, all_mahimahi_ptrs=None, 
                 video_size_dir=None, fixed=False, trace_num=100, **kwargs):
        assert len(all_cooked_time) == len(all_cooked_bw)
        assert trace_num > 0

        np.random.seed(RANDOM_SEED)
        self.fixed = fixed
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.all_file_names = all_file_names
        self.all_mahimahi_ptrs = all_mahimahi_ptrs

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_num = trace_num
        self.all_trace_indices = list(range(len(self.all_cooked_time)))
        if not fixed:
            np.random.shuffle(self.all_trace_indices)

        if self.all_mahimahi_ptrs is None or len(self.all_mahimahi_ptrs) == 0:
            for idx in self.all_trace_indices:
                self.all_mahimahi_ptrs.append(np.random.randint(1, len(self.all_cooked_bw[idx])))
        else:
            self.all_mahimahi_ptrs = [self.all_mahimahi_ptrs[idx] for idx in self.all_trace_indices]
        
        self.trace_indices = self.all_trace_indices[:trace_num]
        self.mahimahi_ptrs = self.all_mahimahi_ptrs[:trace_num]

        self.trace_iter = 0
        self.trace_idx = self.trace_indices[self.trace_iter]
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        # the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_iter = 0
        self.mahimahi_ptr = self.mahimahi_ptrs[self.mahimahi_iter]
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(video_size_dir + 'video_size_' + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE  # throughput = bytes per ms
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration

            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0


        delay *= MILLISECONDS_IN_SECOND

        delay += LINK_RTT

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            self.trace_iter = (self.trace_iter + 1) % self.trace_num
            self.trace_idx = self.trace_indices[self.trace_iter]

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_iter = (self.mahimahi_iter + 1) % self.trace_num
            self.mahimahi_ptr = self.mahimahi_ptrs[self.mahimahi_iter]
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
