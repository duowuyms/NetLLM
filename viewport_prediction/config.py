"""
This file contains the configuration information
"""
import os


class Config:
    dataset_list_360 = ['Wu2017', 'Jin2022']
    dataset_list_vv = ['Serhan2020', 'Hu2023']
    plm_types = ['t5-lm', 'gpt2', 'bert', 'roberta', 't5','llama', 'llava', 'mistral', 'opt']
    plm_sizes = ['small', 'base', 'large', 'xl', 'xll', 'xs', 'xxs']

    # directory and path
    dataset_360 = {
        'Wu2017': '/data/data1/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Wu2017',
        'Jin2022': '/data/data1/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Jin2022'
    }
    dataset_vv = {
        'Serhan2020': '/data/data1/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/Serhan2020',
        'Hu2023': '/data/data1/wuduo/2023_prompt_learning/datasets/viewport_prediction/vv/Hu2023',
    }
    dataset_images_360 = {
        'Jin2022': '/data-HDD/wuduo/2023_prompt_learning/datasets/viewport_prediction/360/Jin2022images/saliencyMap'
    }
    dataset_cache_dir = '/data/data1/wuduo/2023_prompt_learning/datasets/viewport_prediction'
    plms_dir = '/data/data1/wuduo/2023_prompt_learning/downloaded_plms'
    plms_finetuned_dir = '/data-HDD/wuduo//2023_prompt_learning/finetuned_plms/viewport_prediction'
    models_dir = '/data/data1/wuduo/2023_prompt_learning/models/viewport_prediction'
    results_dir = 'viewport_prediction/results' if 'viewport_prediction' not in os.getcwd() else 'results'

    # train, valid, test split info
    dataset_video_split_360 = {
        'Wu2017': {  # proportion: train : valid : test = 10 : 4 : 4
            'train': [9, 7, 4, 1, 6],
            'valid': [8, 2],
            'test': [3, 5],
        },
        'Jin2022': { # proportion: train : valid: test = 5 : 2 : 2
            'train': [1, 5, 9, 2, 6, 11, 15, 16, 13, 17, 21, 22, 26, 19, 23],
            'valid': [3, 7, 12, 10, 20, 27],
            'test': [4, 8, 14, 18, 24, 25]
            # 'test': [4]
        }
    }
    dataset_video_split_vv = {
        'Serhan2020': {  # this dataset only contains one video
            'train': [1],
            'valid': [1],
            'test': [1],
        },
        'Hu2023': {  # proportion: train : valid : test = 4 : 1 : 1
            'train': [1, 2, 3, 6],
            'valid': [4],
            'test': [5]
        }
    }
    dataset_user_split_360 = {
        'Wu2017': {  # proportion: train : valid : test = 30: 9: 9
            'train': [23, 3, 19, 24, 18, 28, 9, 26, 6, 15, 21, 25, 32, 38, 45, 34, 48, 44, 29, 43, 4, 36, 39, 8, 17, 42, 30, 11, 22, 37],
            'valid': [40, 20, 14, 1, 31, 12, 33, 27, 10],
            'test': [16, 7, 46, 5, 35, 2, 41, 47, 13],
        },
        'Jin2022': {  # proportion: train : valid : test = 42: 21: 21
            'train': [50, 54, 6, 34, 66, 63, 52, 39, 62, 46, 75, 28, 65, 18, 37, 13, 80, 33, 69, 78, 19, 40, 10, 43, 61, 72, 56, 41, 79, 82, 27, 71, 57, 67, 8, 2, 12, 81, 1, 64, 32, 42],
            'valid': [9, 25, 73, 29, 31, 70, 58, 11, 14, 38, 16, 76, 77, 74, 24, 5, 17, 20, 51, 68, 36],
            'test': [83, 15, 3, 35, 48, 22, 55, 4, 45, 60, 26, 21, 44, 23, 53, 84, 59, 49, 7, 47, 30]
            # 'test': [83]
        }
    }
    dataset_user_split_vv = {
        'Serhan2020': {  # proportion: train : valid : test = 3 : 1 : 1
            'train': [2, 5, 3],
            'valid': [4],
            'test': [1],
        },
        'Hu2023': {  # this dataset only contains 5 users
            'train': [1, 2, 3, 4, 5],
            'valid': [1, 2, 3, 4, 5],
            'test': [1, 2, 3, 4, 5],
        }
    }
    video_frame = {
        'Wu2017': [30, 30, 30, 30, 30, 30, 25, 25, 30],
        'Jin2022': [30, 30, 30, 30, 30, 30, 30, 30, 30, 25, 25, 25, 25, 25, 25, 25, 25, 25, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    }

    # common default settings
    default_history_window = 10  # historical window
    default_future_window = 20  # future (prediction) window
    default_trim_head = default_history_window * 3  # trim some part of viewport trajectory head
    default_trim_tail = default_future_window * 3  # trim some part of viewport trajectory tail
    default_dataset_frequency = 5  # we have siplify datasets into different frequencies, so we need to specify a frequency for loading the corresponding version of dataset
    default_sample_step = 1  # the step for sampling viewports
    default_epochs = 80  # training epochs
    default_epochs_per_valid = 3  # the number of epochs per validation
    default_steps_per_valid = 500  # the number of steps per validation

    # time-series model default settings
    default_bs = 1  # batch size
    default_grad_accum_step = 1  # gradient accumulation steps
    default_lr = 5e-4  # learning rate
    default_weight_decay = 1e-5  # weight decay
    

cfg = Config()
