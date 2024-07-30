import csv
import numpy as np
from collections import defaultdict
from prettytable import PrettyTable
from utils.metrics import compute_mae, compute_rmse, compute_each_mae, compute_each_rmse


class _PredictionNote:
    def __init__(self, timestep, pred, gt, other=None):
        self.timestep = timestep
        self.pred = pred
        self.gt = gt
        self.other = other


class ResultNotebook:
    """
    A class of notebook to record the prediction results.
    It produces a result.csv file, recording the summary information of the prediction

    The results.csv file structure is illustrated as follows:
    video_id,user_id,mae,rmse
    """
    def __init__(self):
        self.prediction_record = defaultdict(list)
        self.prediction_note = _PredictionNote

        self.position2dimension = {}
        self.position2dimension.update({'roll': 0, 'pitch': 1, 'yaw': 2})

    def record(self, prediction, ground_truth, videos, users, timesteps):
        prediction = prediction.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()
        videos = videos.cpu().numpy()
        users = users.cpu().numpy()
        timesteps = timesteps.cpu().numpy()

        batch_size = prediction.shape[0]
        for i in range(batch_size):
            video, user, timestep = videos[i], users[i], timesteps[i]
            self.prediction_record[video, user].append(self.prediction_note(timestep, prediction[i], ground_truth[i]))

    def write(self, result_path):
        header = ['video', 'user', 'mae', 'rmse', ]
    
        def write_row(writer, pt, video, user, pred, gt):
            if len(pred) == 0 or len(gt) == 0:  
                mae, rmse = float('inf'), float('inf')
            else:
                mae = compute_mae(pred, gt, rotation=True)
                rmse = compute_rmse(pred, gt, rotation=True)
            writer.writerow([video, user, float(mae), float(rmse)])
            pt.add_row([video, user, float(mae), float(rmse)])
            return mae, rmse

        with open(result_path, 'w', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            pretty_table = PrettyTable(field_names=header)

            total_pred, total_gt = [], []
            for video, user in self.prediction_record.keys():
                pred_per_pair, gt_per_pair = [], []
                for note in self.prediction_record[video, user]:
                    pred_per_pair.append(note.pred)
                    gt_per_pair.append(note.gt)
                total_pred.extend(pred_per_pair)
                total_gt.extend(gt_per_pair)

                # compute the mae and rmse for each video user pair
                pred_per_pair = np.array(pred_per_pair)
                gt_per_pair = np.array(gt_per_pair)
                write_row(csv_writer, pretty_table, video, user, pred_per_pair, gt_per_pair)

            # compute the mae and rmse for all video user pairs
            total_pred = np.array(total_pred)
            total_gt = np.array(total_gt)
            # we use -1 to denote the coverage of all pairs
            write_row(csv_writer, pretty_table, -1, -1, total_pred, total_gt)
            print(pretty_table)
            print('Results saved at', result_path)
            file.close()
        
        details_path = result_path.replace('result_', 'details_')
        with open(details_path, 'w') as file:
            for i in range(len(total_pred)):
                pred_line = 'pred: '
                for j in range(len(total_pred[i, 0])):
                    pred_line += f'axis {j + 1}: '
                    pred_line += ', '.join(map(str, total_pred[i, :, j]))
                    pred_line = pred_line[:-1] +'. '
                file.write(pred_line + '\n')
                gt_line = 'gt: '
                for j in range(len(total_gt[i, 0])):
                    gt_line += f'axis {j + 1}: '
                    gt_line += ', '.join(map(str, total_gt[i, :, j]))
                    gt_line = gt_line[:-1] +'. '
                file.write(gt_line + '\n')
            print('Detail results saved at', details_path)
            file.close()

    def write_detail(self, result_path):
        header = ['video', 'user', 'mae', 'rmse', ]
    
        def write_row_detail(writer, pt, video, user, pred, gt):
            if len(pred) == 0 or len(gt) == 0:  
                mae, rmse = float('inf'), float('inf')
            else:
                mae = compute_mae(pred, gt, rotation=True)
                rmse = compute_rmse(pred, gt, rotation=True)
                eachmae = compute_each_mae(pred, gt, rotation=True)
                # print(eachmae)
                eachrmse = compute_each_rmse(pred, gt, rotation=True)
            # writer.writerow([video, user, float(mae), float(rmse)])
            # pt.add_row([video, user, float(mae), float(rmse)])
            for i in range(len(eachmae)):
                writer.writerow([video, user, float(eachmae[i]), float(eachrmse[i])])
                pt.add_row([video, user, float(eachmae[i]), float(eachrmse[i])])
            return mae, rmse

        with open(result_path, 'w', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(header)
            pretty_table = PrettyTable(field_names=header)

            total_pred, total_gt = [], []
            for video, user in self.prediction_record.keys():
                pred_per_pair, gt_per_pair = [], []
                for note in self.prediction_record[video, user]:
                    pred_per_pair.append(note.pred)
                    gt_per_pair.append(note.gt)
                total_pred.extend(pred_per_pair)
                total_gt.extend(gt_per_pair)

                # compute the mae and rmse for each video user pair
                pred_per_pair = np.array(pred_per_pair)
                gt_per_pair = np.array(gt_per_pair)
                write_row_detail(csv_writer, pretty_table, video, user, pred_per_pair, gt_per_pair)

            print(pretty_table)
            print('Results saved at', result_path)
            file.close()

    def reset(self):
        self.prediction_record.clear()
