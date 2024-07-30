"""
Process datasets, including:
1. Convert quaternion to Euler angle
2. Extract key items
3. Simplify datasets
"""
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation


# raw datasets
RAW_DATASETS = {
    'Wu2017': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/raw/Wu2017/Formated_Data',
    'Jin2022': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/raw/Jin2022/origin'
}

# new datasets
NEW_DATASETS = {
    'Wu2017': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Wu2017',
    'Jin2022':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Jin2022'
}

def euler_from_quaternion(quaternion, order='ZXY', degrees=True):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    :param quaternion
    :param order: rotation order
    :param degrees: represent angle as degree (True) or radian (False)
    :return Euler angles
    """
    quaternion = Rotation.from_quat(quaternion)
    euler = quaternion.as_euler(order, degrees=degrees)
    e_roll, e_pitch, e_yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    return e_roll, e_pitch, e_yaw


def process_datasets(dataset='Wu2017'):
    """
    Process datasets: extract key items, convert quaternion to Euler angle
    """
    assert dataset in ['Wu2017', 'Jin2022']

    raw_dataset = RAW_DATASETS[dataset]
    new_dataset = NEW_DATASETS[dataset]   

    # 360Type: Wu2017, Jin2022        

    if dataset == 'Wu2017':
        # Wu2017 dataset contains two parts: Experiment1 and Experiment2, each with 9 videos
        # we name the videos in Experiment1 as 1~9, and those in Experiment2 as 10~18
        video_num, user_num = 9, 48
        for i in range(1, user_num + 1):
            for j in range(1, video_num * 2 + 1):
                raw_data_path = os.path.join(raw_dataset, f'Experiment_{math.ceil(j / video_num)}',
                                             str(i), f'video_{(j - 1) % video_num}.csv')
                raw_data = np.loadtxt(raw_data_path, delimiter=',', usecols=(1, 2, 3, 4, 5), dtype=str)
                raw_data = raw_data[1:, :].astype(np.float32)  # skip headline and convert data into float
                playback_time, quaternion = raw_data[:, 0], raw_data[:, 1:]
                e_roll, e_pitch, e_yaw = euler_from_quaternion(quaternion, 'ZXY', degrees=True)

                new_data = np.stack((playback_time, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{i}.csv')
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')

    elif dataset == 'Jin2022':
        video_num, user_num = 27, 100
        label = 0
        for i in range(1, user_num+1):
            raw_datafile_path = os.path.join(raw_dataset, f'V2 ({i})')
            files = os.listdir(raw_datafile_path)  
            num_png = len(files)
            if ( num_png != 27) or (i == 51):
                continue
            label += 1
            for file in os.listdir(raw_datafile_path):
                j = file.split('_')[2]
                raw_data = np.loadtxt(raw_datafile_path + '/' + file, delimiter=',', usecols=(0,4,5,6,7), dtype=str)       
                raw_data = raw_data[1:, :]
                for line in raw_data:
                    line[1] = line[1][2:]
                    line[4] = line[4][:-2]
                raw_data = raw_data.astype(np.float32)
                playback_time, quaternion = raw_data[:, 0], raw_data[:, 1:]
                e_roll, e_pitch, e_yaw = euler_from_quaternion(quaternion, 'ZXY', degrees=True)
                new_data = np.stack((playback_time, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{label}.csv')
                print(new_data_path)
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')
                

def simplify_datasets(dataset='Wu2017', frequency=5):
    """
    Simplify datasets according to the sample frequency (default 5Hz).
    """
    assert dataset in ['Wu2017', 'Jin2022']

    if dataset == 'Wu2017':
        new_dataset = NEW_DATASETS[dataset]
        video_num, user_num = 18, 48
    elif dataset == 'Jin2022':
        new_dataset = NEW_DATASETS[dataset]
        video_num, user_num = 27, 84

    for i in range(1, user_num + 1):
        for j in range(1, video_num + 1):
            new_data_path = os.path.join(new_dataset, f'video{j}', f'user{i}.csv')
            new_data = np.loadtxt(new_data_path, delimiter=',', dtype=np.float32)
            simplify_data = []
            timestamp, gap = 0, 1 / frequency
            rela_time = new_data[0][0]
            for row in new_data:
                playback_time = ( row[0] - rela_time ) if dataset == 'Jin2022' else row[0]
                if int(playback_time) > 0 and timestamp == 0:  # filter out dirty data
                    continue
                if playback_time >= timestamp:
                    simplify_data.append(row)
                    timestamp += gap
            simplify_data = np.array(simplify_data)
            simplify_data_dir = os.path.join(new_dataset, f'video{j}', f'{frequency}Hz')
            if not os.path.exists(simplify_data_dir):
                os.makedirs(simplify_data_dir)
            simplify_data_path = os.path.join(simplify_data_dir, f'simple_{frequency}Hz_user{i}.csv')
            np.savetxt(simplify_data_path, simplify_data, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    dataset_list = ['Jin2022', 'Wu2017']
    for dataset in dataset_list:
        process_datasets(dataset)
        simplify_datasets(dataset, frequency=5)
