import os
import numpy as np
import sys
import cv2
import pickle
from torch.utils.data import Dataset
from config import cfg


class ViewportDataset(Dataset):
    """
    Wrapper class for viewport dataset.
    """
    def __init__(self, total_traces, total_content_features, videos, users,
                 his_window, fut_window, trim_head, trim_tail, step, for_multi):
        """
        :param total_traces: total viewport traces
        :param total_content_features: total video content features
        :param videos: video list
        :param users: user list
        :param his_window: historical window
        :param fut_window: future (prediction) window
        :param trim_head: trim some part of the viewport trajectory head
        :param trim_tail: trim some part of the viewport trajectory tail
        :param step: step size of sliding prediction window
        """
        self.total_traces = total_traces
        self.total_content_features = total_content_features
        self.videos = videos
        self.users = users
        self.history_window = his_window
        self.future_window = fut_window
        self.trim_head = trim_head
        self.trim_tail = trim_tail
        self.step = step
        self.for_multi = for_multi

        # total_traces store the viewport trace of each video and each user
        # we create a list trace_indices to record the indices to the samples in the traces of specific videos and users
        # the idea here is inspired by Quentin Guimard's repo: https://gitlab.com/DVMS_/DVMS
        self.trace_indices = []
        self.content_feature_indices = []  # TODO
        for video in videos:
            for user in users:
                trace = self.total_traces[video][user]
                for timestep in range(self.trim_head, len(trace) - self.trim_tail, self.step):
                    self.trace_indices.append((video, user, timestep))

        if self.for_multi == True:
            for video in videos:
                image_trace = len(self.total_content_features[video])
                for timestep in range(self.trim_head, image_trace - self.trim_tail, self.step):
                    self.content_feature_indices.append((video,timestep))

    def __len__(self):
        return len(self.trace_indices)

    def __getitem__(self, index):
        """
        With index and self.trace_indices, we can easily access a specific viewport trajectory in the dataset.
        This method is implemented by subclass ViewportDataset360 and ViewportDatasetVV.
        """
        if self.for_multi == False:
            video, user, timestep = self.trace_indices[index]
            history = self.total_traces[video][user][timestep - self.history_window:timestep]
            future = self.total_traces[video][user][timestep:timestep + self.future_window]
            return history, future, (video, user, timestep)
        
        else:
            video, user, timestep = self.trace_indices[index]
            history = self.total_traces[video][user][timestep - self.history_window:timestep]
            future = self.total_traces[video][user][timestep:timestep + self.future_window]
            history_images = []
            future_images = []
            his_index_start = timestep - self.history_window
            fut_index_start = his_index_end = timestep
            fut_index_end  = self.future_window + fut_index_start

#             print(f'''his_start: {his_index_start}
# his_index_end: {his_index_end}
# fut_index_start: {fut_index_start}
# fut_index_end: {fut_index_end}''')
            # import IPython; IPython.embed()?
            
            for c in range(his_index_start,his_index_end):
                history_images.append(self.total_content_features[video][c])
            for c in range(fut_index_start,fut_index_end):
                future_images.append(self.total_content_features[video][c])

            return history, future, history_images, future_images, (video, user, timestep)


class ViewportDatasetForPrompt(ViewportDataset):
    """
    Wrapper class for ViewportDataset for Prompt Learning pipeline.
    """
    def __init__(self, total_traces, total_content_features, videos, users,
                 his_window, fut_window, trim_head, trim_tail, step, precision, delimiter, dataset, dataset_type):
        super().__init__(total_traces, total_content_features, videos, users, 
                         his_window, fut_window, trim_head, trim_tail, step)
        self.precision = precision  # precision for significant figures
        self.delimiter = delimiter
        self.dataset = dataset
        self.dataset_type = dataset_type  

    def __getitem__(self, index):
        """
        A small modifications on the base __getitem__ method.
        The return viewports are transformed into strings
        """
        video, user, timestep = self.trace_indices[index]
        history = self.total_traces[video][user][timestep - self.history_window:timestep]
        # print(history)
        history = normalize_data(history, self.dataset, self.dataset_type)
        # print(history_normalized)
        future = self.total_traces[video][user][timestep:timestep + self.future_window]
        num_axes = history.shape[-1]
        history_info = {'ts_start': str(timestep - self.history_window + 1), 'ts_end': str(timestep), 'num_axes': str(num_axes)}
        future_info = {'ts_start': str(timestep + 1), 'ts_end': str(timestep + self.future_window), 'num_axes': str(num_axes)}
        for i in range(num_axes):
            history_info[f'axis {i + 1}'] = self.delimiter.join(
                [np.format_float_positional(history[j, i], precision=self.precision, unique=False, fractional=False, trim='0')
                 for j in range(self.history_window)]
            )
            future_info[f'axis {i + 1}'] = self.delimiter.join(
                [np.format_float_positional(future[j, i], precision=self.precision,  unique=False, fractional=False, trim='0') 
                 for j in range(self.future_window)]
            )
        return history_info, future_info, (video, user, timestep), future


class ViewportDatasetForPromptMultimodality(ViewportDataset):
    pass

def pack_data(dataset_dir, video_user_pairs, frequency, for_multi, dataset):
    """
    Pack the viewport traces and video content features of corresponding video and user pairs
    into easy-access dict objects
    :param dataset_dir: directory of dataset
    :param video_user_pairs: list of video-user pairs
    :param frequency: the frequency version of the dataset
    :return: total_traces, total_content_features
    """
    pack_traces = {video: {} for video, _ in video_user_pairs}
    pack_content_features = {video: {} for video, _ in video_user_pairs}
    if for_multi == False:
        for video, user in video_user_pairs:
            data_path = os.path.join(dataset_dir, f'video{video}', f'{frequency}Hz', f'simple_{frequency}Hz_user{user}.csv')
            data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)
            pack_traces[video][user] = data[:, 1:]  # the first column (i.e., column = 0) is timestep, we don't need it
    if for_multi == True:
        # image_data_total_path = cfg.dataset_images_360['Wu2017']
        # image_data_total_path = cfg.dataset_images_360['Jin2022']
        image_data_total_path = cfg.dataset_images_360[dataset]
        for video, user in video_user_pairs:
            # print("\rfor debug",video, user)

            image_data_path = os.path.join(image_data_total_path, f'video{video}_images')
            data_path = os.path.join(dataset_dir, f'video{video}', f'{frequency}Hz', f'simple_{frequency}Hz_user{user}.csv')
            tmp_data = np.loadtxt(data_path, delimiter = ',', dtype=np.float32)
            pack_traces[video][user] = tmp_data[:, 1:] 

            if len(pack_content_features[video]) > 0:
                continue
            
            if dataset == 'Jin2022':
                if video in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
                    total_images = 1500
                else:
                    total_images = 1800
            if dataset == 'Wu2017':
                total_images = cfg.Wu2017_video_image[video-1]
            image_freq = int(total_images/len(tmp_data[:, 0]))
            image_names = []
            
            for k in range(1,total_images + 1):
                if ((k-1) % image_freq == 0):
                    image_names.append(os.path.join(image_data_path, f'{k}.png'))
            
            c = 1
            pre_image = None
            for image_name in image_names:
                # print("image_name:", image_name )
                # if c <= len(tmp_data[:, 0]):
                if os.path.exists(image_name):
                    image = cv2.imread(image_name)
                    image = cv2.resize(image, (224, 224))
                    # print(image_name)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # CAN IT?
                else:
                    gray_image = pre_image
                pack_content_features[video][c] = gray_image
                c += 1
                pre_image = gray_image

    return pack_traces, pack_content_features


def create_dataset(dataset, dataset_type, dataset_video_split=None, dataset_user_split=None,
                   dataset_image=None, his_window=cfg.default_history_window, fut_window=cfg.default_future_window,
                   trim_head=cfg.default_trim_head, trim_tail=cfg.default_trim_tail, 
                   frequency=cfg.default_dataset_frequency, step=cfg.default_sample_step,
                   include=('train', 'valid', 'test'), precision=cfg.default_precision, 
                   delimiter=cfg.default_delimiter, for_prompt=False, for_multi=False):
    """
    Create dataset.
    :param dataset: dataset name
    :param dataset_type: dataset type (360 or vv)
    :param dataset_video_split: train, valid, test split info of videos
    :param dataset_user_split: train, valid, test split info of users
    :param his_window: historical window
    :param fut_window: future (prediction) window
    :param trim_head: trim some part of the viewport trajectory head
    :param trim_tail: trim some part of the viewport trajectory tail
    :param frequency: we have simplify datasets into different frequencies, so we need to specify a frequency to load the coresponding version of dataset
    :param step:the step for sampling viewports
    :param include: inclusion of the splits of dataset
    :param precision: the precision to round viewport positions into significant figures (only valid when for_prompt=True)
    :param delimiter: the delimiter to separate different viewport positions (only valid when for_prompt=True)
    :param for_prompt: whether the dataset is for prompt learning
    :return: dataset_train, dataset_valid, dataset_test
    """
    if dataset_type == '360':
        dataset_dir = cfg.dataset_360[dataset]
        if dataset_video_split is None:
            dataset_video_split = cfg.dataset_video_split_360[dataset]
        if dataset_user_split is None:
            dataset_user_split = cfg.dataset_user_split_360[dataset]
        # if for_multi == True and dataset_image is None:
        #     dataset_image_dir = cfg.dataset_images_360
            #image split is the same as dataset_video_split
    else:
        dataset_dir = cfg.dataset_vv[dataset]
        if dataset_video_split is None:
            dataset_video_split = cfg.dataset_video_split_vv[dataset]
        if dataset_user_split is None:
            dataset_user_split = cfg.dataset_user_split_vv[dataset]

    total_video_user_pairs = []
    for split in include:
        videos = dataset_video_split[split]
        users = dataset_user_split[split]
        for video in videos:
            for user in users:
                total_video_user_pairs.append((video, user))
    total_traces, total_content_features = pack_data(dataset_dir, total_video_user_pairs, frequency, for_multi, dataset)
    dataset_splits = []
    for split in include:
        if not for_prompt:
            # print('[cp>> load_dataset.py:225]')
            # import IPython; IPython.embed()
            dataset_splits.append(
                ViewportDataset(total_traces, total_content_features, dataset_video_split[split],
                                dataset_user_split[split], his_window, fut_window, trim_head, trim_tail, step, for_multi)
            )
        else:
            dataset_splits.append(
                ViewportDatasetForPrompt(total_traces, total_content_features, dataset_video_split[split],
                                         dataset_user_split[split], his_window, fut_window, trim_head, trim_tail, 
                                         step, precision, delimiter, dataset, dataset_type)
            )
    return dataset_splits


def _test_create_dataset():
    dataset = 'Jin2022'
    dataset_type = '360'
    dataset_video_split = {'train': [1], 'valid': [2], 'test': [3]}
    dataset_user_split = {'train': [1], 'valid': [2], 'test': [3]}
    dataset_train, *_ = create_dataset(dataset, dataset_type, dataset_video_split, dataset_user_split, for_prompt=False, for_multi=True)
    print(len(dataset_train), '\n')
    print('======== 360 =======')
    print(dataset_train[1])
    
    # his, fut, _ = dataset_train[0]
    # print(his)
    # print(fut, '\n')

    # dataset = 'Wu2017'
    # dataset_type = '360'
    # dataset_video_split = {'train': [1], 'valid': [2], 'test': [3]}
    # dataset_user_split = {'train': [1], 'valid': [2], 'test': [3]}
    # dataset_train, *_ = create_dataset(dataset, dataset_type, dataset_video_split, dataset_user_split)
    # print(len(dataset_train), '\n')
    # print('======== 360-1 =======')
    # for i in range(5):
    #     his, fut = dataset_train[i]
    #     print(his)
    #     print(fut, '\n')
    # print('======== 360-2 =======')
    # for i in range(-5, 0):
    #     his, fut = dataset_train[i]
    #     print(his)
    #     print(fut, '\n')

    # dataset = 'Serhan2020'
    # dataset_type = 'vv'
    # dataset_video_split = {'train': [1], 'valid': [1], 'test': [1]}
    # dataset_user_split = {'train': [1], 'valid': [2], 'test': [3]}
    # dataset_train, *_ = create_dataset(dataset, dataset_type, dataset_video_split, dataset_user_split)
    # print(len(dataset_train), '\n')
    # print('======== vv-1 =======')
    # for i in range(5):
    #     his_trans, fut_trans, his_rot, fut_rot = dataset_train[i]
    #     print('trans', his_trans)
    #     print(fut_trans, '\n')
    #     print('rot', his_rot)
    #     print(fut_rot, '\n')
    # print('======== vv-2 =======')
    # for i in range(-5, 0):
    #     his_trans, fut_trans, his_rot, fut_rot = dataset_train[i]
    #     print('trans', his_trans)
    #     print(fut_trans, '\n')
    #     print('rot', his_rot)
    #     print(fut_rot, '\n')


def _test_create_dataset_for_prompt():
    dataset = 'Jin2022'
    dataset_type = '360'
    dataset_video_split = {'train': [5], 'valid': [8], 'test': [9]}
    dataset_user_split = {'train': [5], 'valid': [8], 'test': [9]}
    dataset_train, *_ = create_dataset(dataset, dataset_type, dataset_video_split, dataset_user_split, for_prompt=False, for_multi=False)

    print(len(dataset_train), '\n')
    print('======== 360 =======')
    his, fut, _ = dataset_train[0]
    print(his)
    print(fut, '\n')


    # dataset = 'Serhan2020'
    # dataset_type = 'vv'
    # dataset_video_split = {'train': [1], 'valid': [1], 'test': [1]}
    # dataset_user_split = {'train': [1], 'valid': [2], 'test': [3]}
    # dataset_train, *_ = create_dataset(dataset, dataset_type, dataset_video_split, dataset_user_split, for_prompt=True)
    # print(len(dataset_train), '\n')
    # print('======== vv =======')
    # his, fut, _ = dataset_train[0]
    # print(his)
    # print(fut, '\n')


if __name__ == '__main__':
    _test_create_dataset()
    #_test_create_dataset_for_prompt()
