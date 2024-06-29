MODEL_SAVE_INTERVAL = 500
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
MAX_VIDEO_BIT_RATE = VIDEO_BIT_RATE[-1]
HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent

BITRATE_LEVELS = 6


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement (throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 6  # take how many frames in the past
# jump-action dim
A_DIM = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
BUFFER_NORM_FACTOR = 10.0
RAND_RANGE = 1000
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size
# download_time reward
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
VIDEO_CHUNK_LEN = 4000.0  # millisec, every time add this amount to buffer
TOTAL_VIDEO_CHUNK = 48.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
TRAIN_SEQ_LEN = 100  # batchsize of pensieve training
VIDEO_SIZE_FILE = '/home/wuduo/notmuch/projects/Genet/data/video2_sizes/video_size_'

