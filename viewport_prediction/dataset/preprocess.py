"""
Process datasets, including:
1. Convert quaternion to Euler angle
2. Extract key items
3. Simplify datasets
"""
import math
import os
import numpy as np
import pandas as pd
from random import randint
from scipy.spatial.transform import Rotation
from prettytable import PrettyTable


# raw datasets
RAW_DATASETS_360 = {
    'Wu2017': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/raw/Wu2017/Formated_Data',
    'Jin2022': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/raw/Jin2022/origin'
}
RAW_DATASETS_VV = {
    'Serhan2020': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/raw/Serhan2020',
    'Hu2023':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/raw/Hu2023',
    'Hu2023full':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/raw/Hu2023full'
}
# new datasets
NEW_DATASETS_360 = {
    'Wu2017': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Wu2017',
    'Jin2022':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Jin2022'
}
NEW_DATASETS_VV = {
    'Serhan2020': '/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/Serhan2020',
    'Hu2023':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/Hu2023',
    'Hu2023full':'/data2/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/Hu2023full'
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


def process_datasets(dataset='Wu2017', dataset_type='360'):
    """
    Process datasets: extract key items, convert quaternion to Euler angle
    """
    assert dataset in ['Wu2017', 'Serhan2020', 'Hu2023', 'Jin2022', 'Hu2023full']
    assert dataset_type in ['360', 'vv']

    if dataset_type == '360':
        raw_dataset = RAW_DATASETS_360[dataset]
        new_dataset = NEW_DATASETS_360[dataset]   
    elif dataset_type == 'vv':
        raw_dataset = RAW_DATASETS_VV[dataset]
        new_dataset = NEW_DATASETS_VV[dataset]    

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

    # VVtype: Serhan2020, Hu2023, Hu2023full
    elif dataset == 'Serhan2020':
        # Serhan2020 only contains 1 video with 5 users
        video_num, user_num = 1, 5
        for i in range(1, user_num + 1):
            for j in range(1, video_num + 1):
                raw_data_path = os.path.join(raw_dataset, f'User{i}.csv')
                raw_data = np.loadtxt(raw_data_path, delimiter=',', usecols=(0, 1, 2, 3, -3, -2, -1), dtype=str)
                raw_data = raw_data[1:, :].astype(np.float32)
                playback_time = raw_data[:, 0] / 10 ** 9  # convert ns -> s
                x, y, z = raw_data[:, 1], raw_data[:, 2], raw_data[:, 3]
                e_roll, e_pitch, e_yaw = raw_data[:, 4], raw_data[:, 5], raw_data[:, 6]

                new_data = np.stack((playback_time, x, y, z, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{i}.csv')
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')

    elif dataset == 'Hu2023':
        video_num, user_num = 6, 5
        video_name_dict = {  
            1 : "chatting" ,
            2 : "cleaning_whiteboard" ,
            3 : "News_interviewing" ,
            4 : "presenting" ,
            5 : "Pulling_trolley" ,
            6 : "sweep"
        }
        user_name_dict = {
            1 : "fupingyu" ,
            2 : "Guozhaonian" ,
            3 : "huangrenyi" , 
            4 : "liuxuya" ,
            5 : "Sunqiran"
        }
        for i in range(1, user_num + 1):
            user_name = user_name_dict[i]
            for j in range(1, video_num + 1):
                video_name = video_name_dict[j]
                raw_data_path = os.path.join(raw_dataset, f'{user_name}_{video_name}.txt')
                raw_data =  np.loadtxt(raw_data_path, delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7), dtype=str)
                
                raw_data = raw_data.astype(np.float32)
                time = raw_data[:, 0]
                x, y, z = raw_data[:, 1], raw_data[:, 2], raw_data[:, 3]
                e_roll, e_pitch, e_yaw = raw_data[:, 6], raw_data[:, 5], raw_data[:, 4]
                new_data = np.stack((time, x, y, z, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{i}.csv')
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')

    elif dataset == 'Hu2023full':
        video_num, user_num = 6, 13
        video_name_dict = {  
            "chatting" : 1,
            "cleaning_whiteboard" : 2,
            "News_interviewing" : 3,
            "presenting" : 4,
            "Pulling_trolley" : 5,
            "sweep" : 6
        }
        user_name_dict = {
            "ChenYongting" : 1, 
            "GuoYushan" : 2,
            "RenZhichen" : 3,
            "WangYan" : 4,
            "fupingyu" : 5,
            "Guozhaonian" : 6,
            "huangrenyi" : 7,
            "liuxuya" : 8,
            "sulehan" : 9,
            "Sunqiran" : 10,
            "yuchen" :11
            # "TuYuzhao" : 12,
            # "FengXuanqi" : 13
        }

        path1 = os.path.join(raw_dataset, '4.21')
        path2 = os.path.join(raw_dataset, 'vv-ub-04.26')
        
        for file in os.listdir(path1) :
            file_list = file.split('_',1)
            if ( len(file_list) == 2 ) and ( file_list[0] in user_name_dict ) and ( file.endswith('txt') ):
                print(file)
                i = user_name_dict[file_list[0]]
                j = video_name_dict[file_list[1].split('.')[0]]
                raw_data_path = os.path.join(path1,file)
                raw_data =  np.loadtxt(raw_data_path, delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7), dtype=str)
                raw_data = raw_data.astype(np.float32)
                time = raw_data[:, 0]
                x, y, z = raw_data[:, 1], raw_data[:, 2], raw_data[:, 3]
                e_roll, e_pitch, e_yaw = raw_data[:, 6], raw_data[:, 5], raw_data[:, 4]
                new_data = np.stack((time, x, y, z, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{i}.csv')
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')
                
        for file in os.listdir(path2) :
            file_list = file.split('_',1)
            if ( len(file_list) == 2 ) and ( file_list[0] in user_name_dict ) and ( file.endswith('txt') ):
                #print(file)
                i = user_name_dict[file_list[0]]
                j = video_name_dict[file_list[1].split('.')[0]]
                raw_data_path = os.path.join(path2,file)
                raw_data =  np.loadtxt(raw_data_path, delimiter=' ', usecols=(1, 2, 3, 4, 5, 6, 7), dtype=str)
                raw_data = raw_data.astype(np.float32)
                time = raw_data[:, 0]
                x, y, z = raw_data[:, 1], raw_data[:, 2], raw_data[:, 3]
                e_roll, e_pitch, e_yaw = raw_data[:, 6], raw_data[:, 5], raw_data[:, 4]
                new_data = np.stack((time, x, y, z, e_roll, e_pitch, e_yaw), axis=1)
                new_data_dir = os.path.join(new_dataset, f'video{j}')
                if not os.path.exists(new_data_dir):
                    os.makedirs(new_data_dir)
                new_data_path = os.path.join(new_data_dir, f'user{i}.csv')
                np.savetxt(new_data_path, new_data, fmt='%.6f', delimiter=',')
                

def simplify_datasets(dataset='Wu2017', dataset_type='360', frequency=5):
    """
    Simplify datasets according to the sample frequency (default 5Hz).
    """
    assert dataset in ['Wu2017', 'Serhan2020', 'Hu2023', 'Hu2023full', 'Jin2022' ]
    assert dataset_type in ['360', 'vv']

    if dataset == 'Wu2017':
        new_dataset = NEW_DATASETS_360[dataset]
        video_num, user_num = 18, 48
    elif dataset == 'Serhan2020':
        new_dataset = NEW_DATASETS_VV[dataset]
        video_num, user_num = 1, 5
    elif dataset == 'Hu2023':
        new_dataset = NEW_DATASETS_VV[dataset]
        video_num, user_num = 6, 5
    elif dataset == 'Hu2023full':
        new_dataset = NEW_DATASETS_VV[dataset]
        video_num, user_num = 6, 11       
    elif dataset == 'Jin2022':
        new_dataset = NEW_DATASETS_360[dataset]
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


def observe_datasets(video360=True, videovv=True):
    """
    We want to simplify dataset by sampling viewports according to specific frequency (e.g., 5Hz).
    Before that, we need to observe the datasets.
    For 360 video, we use Wu2017 dataset for observation
    For volumetric video, we use Serhan2020 dataset for observation
    """
    if video360:
        # we set 9 candidate frequency: 90, 45, 30, 15, 10, 5, 3, 2, 1
        # in particular, 90 is the original frequency of Wu2017 dataset
        cand_freq = [90, 45, 30, 15, 10, 5, 3, 2, 1]
        origin_freq = 90
        pitch_dict = {freq: [] for freq in cand_freq}  # record the pitch difference of each freq.
        yaw_dict = {freq: [] for freq in cand_freq}  # record the yaw difference of each freq.
        for _ in range(10000):
            data360_path = os.path.join(NEW_DATASETS_360['Wu2017'], f'video{randint(1, 18)}', f'user{randint(1, 48)}.csv')
            data360 = np.loadtxt(data360_path, delimiter=',', dtype=np.float32)
            start = np.random.randint(origin_freq * 5, len(data360) - origin_freq * 5)
            end = start + origin_freq
            for freq in cand_freq:
                step = origin_freq // freq
                for i in range(start, end, step):
                    pitch_dict[freq].append(abs(data360[i + step, -2] - data360[i, -2]))
                    yaw_dict[freq].append(abs(data360[i + step, -1] - data360[i, -1]))
        print('========= 360 Video =========')

    if videovv:
        # we set 9 candidate frequency: 200, 100, 50, 25, 10, 5, 3, 2, 1
        # in particular, 200 is the original frequency of Serhan2020 dataset
        cand_freq = [200, 100, 50, 25, 10, 5, 3, 2, 1]
        origin_freq = 200
        x_dict = {freq: [] for freq in cand_freq}  # record the x_pos difference of each freq.
        y_dict = {freq: [] for freq in cand_freq}  # record the y_pos difference of each freq.
        z_dict = {freq: [] for freq in cand_freq}  # record the z_pos difference of each freq.
        pitch_dict = {freq: [] for freq in cand_freq}  # record the pitch difference of each freq.
        yaw_dict = {freq: [] for freq in cand_freq}  # record the yaw difference of each freq.
        for _ in range(10000):
            datavv_path = os.path.join(NEW_DATASETS_VV['Serhan2020'], 'video1', f'user{randint(1, 5)}.csv')
            datavv = np.loadtxt(datavv_path, delimiter=',', dtype=np.float32)
            start = np.random.randint(origin_freq * 5, len(datavv) - origin_freq * 5)
            end = start + origin_freq
            for freq in cand_freq:
                step = origin_freq // freq
                for i in range(start, end, step):
                    x_dict[freq].append(abs(datavv[i + step, 1] - datavv[i, 1]))
                    y_dict[freq].append(abs(datavv[i + step, 2] - datavv[i, 2]))
                    z_dict[freq].append(abs(datavv[i + step, 3] - datavv[i, 3]))
                    pitch_dict[freq].append(abs(datavv[i + step, -2] - datavv[i, -2]))
                    yaw_dict[freq].append(abs(datavv[i + step, -1] - datavv[i, -1]))
        print('========= Volumetric Video =========')
        pt = PrettyTable()
        pt.field_names = ['Frequency', 'Mean X Diff', 'Mean Y Diff', 'Mean Z Diff', 'Mean Pitch Diff', 'Mean Yaw Diff']
        for freq in cand_freq:
            mean_x_diff = np.mean(x_dict[freq])
            mean_y_diff = np.mean(y_dict[freq])
            mean_z_diff = np.mean(z_dict[freq])
            mean_pitch_diff = np.mean(pitch_dict[freq])
            mean_yaw_diff = np.mean(yaw_dict[freq])
            pt.add_row([freq, mean_x_diff, mean_y_diff, mean_z_diff, mean_pitch_diff, mean_yaw_diff])
        print(pt)
    '''
    Results preview:
    "========= 360 Video =========
    +-----------+-----------------+---------------+
    | Frequency | Mean Pitch Diff | Mean Yaw Diff |
    +-----------+-----------------+---------------+
    |     90    |   0.056125227   |   0.22876297  |
    |     45    |    0.10414556   |   0.4543842   |
    |     30    |    0.1513652    |   0.67920935  |
    |     15    |    0.28837207   |   1.3477213   |
    |     10    |    0.42072386   |    2.006005   |
    |     5     |    0.79582983   |   3.8884444   |
    |     3     |    1.2441044    |    6.193477   |
    |     2     |    1.7431102    |    8.952888   |
    |     1     |    2.9025724    |   15.8131895  |
    +-----------+-----------------+---------------+
    ========= Volumetric Video =========
    +-----------+--------------+---------------+--------------+-----------------+---------------+
    | Frequency | Mean X Diff  |  Mean Y Diff  | Mean Z Diff  | Mean Pitch Diff | Mean Yaw Diff |
    +-----------+--------------+---------------+--------------+-----------------+---------------+
    |    200    | 0.0011076288 | 0.00032912366 | 0.0009540163 |   0.082038745   |   0.23926522  |
    |    100    | 0.0022143922 | 0.00065538887 | 0.0019070352 |    0.16334964   |    0.477956   |
    |     50    | 0.004426181  |  0.0013021705 | 0.0038112653 |    0.32370943   |   0.9536003   |
    |     25    |  0.00884569  |  0.0025727162 | 0.007615154  |    0.6326218    |    1.896818   |
    |     10    | 0.022065034  |  0.0061486773 | 0.018992078  |    1.4688491    |   4.6605964   |
    |     5     |  0.04388905  |  0.011124906  | 0.037755545  |    2.6721156    |     9.0402    |
    |     3     |  0.07180457  |   0.01620997  | 0.061736494  |     4.056462    |   14.098234   |
    |     2     |  0.10674588  |  0.021148702  |  0.09175303  |    5.4506664    |    20.64043   |
    |     1     |  0.20051245  |   0.03282802  |  0.1721155   |     7.875772    |   36.209816   |
    +-----------+--------------+---------------+--------------+-----------------+---------------+"
    '''


if __name__ == '__main__':
    # dataset_list = ['Wu2017', 'Serhan2020']
    dataset_list = ['Jin2022']
    dataset_type = {'Wu2017': '360', 'Serhan2020': 'vv', 'Hu2023': 'vv', 'Hu2023full': 'vv', 'Jin2022' : '360'}
    # observe_dataset(video360=True, videovv=True)
    for dataset in dataset_list:
        process_datasets(dataset, dataset_type[dataset])
        simplify_datasets(dataset, dataset_type[dataset], frequency=5)
