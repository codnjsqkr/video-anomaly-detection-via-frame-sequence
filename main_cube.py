import sys
import os
import pathlib
import argparse
import datetime
import numpy as np

from models import *
import time
import torch
import torchvision
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.optim
import torch.nn.functional as F
from utils.factory import MultiStepLR, PolyLR, CosineAnnealingLR

from utils2 import load_images_and_resize, split_cubes, load_all_cubes_in_set, count_sequence_n_frame
from utils2 import calc_score_full_clips, plot_error_map, load_ground_truth_Boat, load_ground_truth_Avenue
from utils2 import get_test_frame_labels, full_assess_AUC, manual_assess_AUC, full_assess_AUC_multiple_scale
# from utils2 import write_video_result, write_example, visualize_filters, get_weights, convert_model
# from net_model import train_model_naive_with_batch_norm, test_model_naive_with_batch_norm

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU 번호 설정, GPU 한개면 "0"

# dataset
ped2 = {'name': 'ped2',
          'path': '/home/mvp01/chaewon/anomaly22/datasets/ped2',
          'n_clip_train': 16,
          'n_clip_valid': 13,
          'n_clip_test': 12,
          'ground_truth': [[61, 180], [95, 180], [1, 146], [31, 180], [1, 129], [1, 159],
                    [46, 180], [1, 180], [1, 120], [1, 150], [1, 180], [88, 180]],
          'ground_truth_mask': np.arange(12)+1}

Avenue = {'name': 'Avenue',
          'path': '/home/mvp01/chaewon/anomaly22/datasets/avenue',
          'test_mask_path': '/home/mvp01/chaewon/anomaly22/datasets/avenue',
          'n_clip_train': 16,
          'n_clip_test': 21,
          'ground_truth': None,
          'ground_truth_mask': np.arange(21)+1}

dataset_dict = {'ped2': ped2, 'Avenue': Avenue}

'''======================== MAIN ======================='''


def main(argv):
    # set constant
    cube_size = [10, 10, 4]

    #
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='dataset', default='ped2')
    parser.add_argument('-g', '--height', help='frame height', default=120)
    parser.add_argument('-w', '--width', help='frame width', default=160)
    parser.add_argument('-t', '--task', help='task to perform', default=1)
    parser.add_argument('-c', '--clip', help='clip index (zero-based)', default=0)
    parser.add_argument('-s', '--set', help='test set', default=1)
    parser.add_argument('-e', '--epoch', help='epoch destination', default=20)
    parser.add_argument('-m', '--model', help='start model idx', default=0)
    args = vars(parser.parse_args())
    #
    dataset = dataset_dict[args['dataset']]
    dataset['path_train'] = '%s/training' % dataset['path']
    dataset['path_test'] = '%s/testing' % dataset['path']
    #
    task = int(args['task'])
    h = int(args['height'])
    w = int(args['width'])
    clip_idx = int(args['clip'])
    test_set = bool(int(args['set']))
    n_epoch_destination = int(args['epoch'])
    model_idx_to_start = int(args['model'])
    model_test = model_idx_to_start
    n_row, n_col = np.array([h, w]) // cube_size[:2]
    print('Selected task = %d' % task)
    print('started time: %s' % datetime.datetime.now())
    #
    dataset['cube_dir'] = './training_saver/%s/cube_%d_%d_%d_%d_%d' % (dataset['name'], h, w, cube_size[0], cube_size[1], cube_size[2])
    if not os.path.exists(dataset['cube_dir']):
        pathlib.Path(dataset['cube_dir']).mkdir(parents=True, exist_ok=True)

    '''========================================='''
    ''' Task 1: Resize frame resolution dataset '''
    '''========================================='''
    if task == 1:
        load_images_and_resize(dataset, new_size=[h, w], train=True, force_recalc=False, return_entire_data=False)
        load_images_and_resize(dataset, new_size=[h, w], train=False, force_recalc=False, return_entire_data=False)

    '''========================================='''
    ''' Task 2: Split cubes in dataset and save '''
    '''========================================='''
    if task == 2:
        split_cubes(dataset, clip_idx, cube_size, training_set=True, force_recalc=False, dist_thresh=None)

    '''=========================================='''
    ''' Task 3: Train model and check validation '''
    '''=========================================='''
    if task == 3:
        training_cubes, training_mapping, training_seqrot = load_all_cubes_in_set(dataset, h, w, cube_size, training_set=True)
#         valid_cubes, valid_mapping, valid_seqrot = load_all_cubes_in_set(dataset, h, w, cube_size, training_set=False, val=True, test=False)
#         train_model_naive_with_batch_norm(dataset, training_cubes, training_mapping[:, 2], training_mapping[:, 3], n_row, n_col,
#                                           n_epoch_destination, start_model_idx=model_idx_to_start, batch_size=256*12)

    '''====================================='''
    ''' Task 4: Test model and save outputs '''
    '''====================================='''
    if task == 4:
        sequence_n_frame = count_sequence_n_frame(dataset, test=test_set)
        test_cubes, test_mapping = split_cubes(dataset, clip_idx, cube_size, training_set=False)
        test_model_naive_with_batch_norm(dataset, test_cubes, test_mapping[:, 2], test_mapping[:, 3], n_row, n_col,
                                         sequence_n_frame, clip_idx, model_idx=model_test, batch_size=256*12, using_test_data=test_set)

    '''====================================='''
    ''' Task 5: Calculate scores of dataset '''
    '''====================================='''
    if task == 5:
        calc_score_full_clips(dataset, np.array([h, w]), cube_size, model_test, train=False)
        calc_score_full_clips(dataset, np.array([h, w]), cube_size, model_test, train=True)

    '''========================='''
    ''' Task -5: Plot error map '''
    '''========================='''
    if task == -5:
        frame_idx = np.arange(16)
        print('selected set:', 'Test' if test_set else 'Train')
        print('selected frames:', frame_idx)
        plot_error_map(dataset, np.array([h, w]), cube_size, clip_idx, frame_idx, model_test, score_type_idx=3, using_test_data=test_set)

    '''===================='''
    ''' Task 6: Evaluation '''
    '''===================='''
    if task == 6:
        if dataset in [Belleview, Train]:
            dataset['ground_truth'] = load_ground_truth_Boat(dataset, n_clip=dataset['n_clip_test'])
        elif dataset == Avenue:
            dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        labels_select_last, labels_select_first, labels_select_mid = get_test_frame_labels(dataset, sequence_n_frame, cube_size, is_subway=False)
        #
        for way in range(6):
            # sequence_n_frame = None
            if way != 1:
                continue
            op = np.std
            full_assess_AUC(dataset, np.array([h, w]), cube_size, model_test, labels_select_first,
                            sequence_n_frame=sequence_n_frame, plot_pr_idx=None, selected_score_estimation_way=way, operation=op, save_roc_pr=True)

    '''============================'''
    ''' Task -6: Manual evaluation '''
    '''============================'''
    if task == -6:
        if dataset in [Belleview, Train]:
            dataset['ground_truth'] = load_ground_truth_Boat(dataset, n_clip=dataset['n_clip_test'])
        elif dataset == Avenue:
            dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        labels_select_last, labels_select_first, labels_select_mid = get_test_frame_labels(dataset, sequence_n_frame, cube_size, is_subway=False)
        #
        for way in range(6):
            # sequence_n_frame = None
            if way != 1:
                continue
            op = np.std
            #
            manual_assess_AUC(dataset, np.array([h, w]), cube_size, model_test, labels_select_mid,
                              plot_pr_idx=None, selected_score_estimation_way=way, operation=op)
            #
            manual_assess_AUC(dataset, np.array([h, w]), cube_size, model_test, labels_select_first,
                              plot_pr_idx=None, selected_score_estimation_way=way, operation=op)
            #
            manual_assess_AUC(dataset, np.array([h, w]), cube_size, model_test, labels_select_last,
                              plot_pr_idx=None, selected_score_estimation_way=way, operation=op)

    '''==================================='''
    ''' Task 7: Multiple scale evaluation '''
    '''==================================='''
    if task == 7:
        if dataset in [Belleview, Train]:
            dataset['ground_truth'] = load_ground_truth_Boat(dataset, n_clip=dataset['n_clip_test'])
        elif dataset == Avenue:
            dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        labels_select_last, labels_select_first, labels_select_mid = get_test_frame_labels(dataset, sequence_n_frame, cube_size)
        #
        for way in range(6):
            # sequence_n_frame = None
            if way != 1:
                continue
            op = np.std
            #
            full_assess_AUC_multiple_scale(dataset, [np.array([120, 160]), np.array([30, 40]), np.array([20, 20])],
                                           cube_size, model_test, labels_select_mid, sequence_n_frame=sequence_n_frame,
                                           selected_score_estimation_way=way, operation=op)
            #
            full_assess_AUC_multiple_scale(dataset, [np.array([120, 160]), np.array([30, 40]), np.array([20, 20])],
                                           cube_size, model_test, labels_select_first, sequence_n_frame=sequence_n_frame,
                                           selected_score_estimation_way=way, operation=op)
            #
            full_assess_AUC_multiple_scale(dataset, [np.array([120, 160]), np.array([30, 40]), np.array([20, 20])],
                                           cube_size, model_test, labels_select_last, sequence_n_frame=sequence_n_frame,
                                           selected_score_estimation_way=way, operation=op)

    '''========================='''
    ''' Task 08: Write video    '''
    ''' Task 11: Save score plot'''
    '''========================='''
    if task == 8 or task == 11:
        frame_ranges = {'Belleview': (50, 443+157), 'Train': (2100, 3200)}
        if dataset in [Belleview, Train]:
            dataset['ground_truth'] = load_ground_truth_Boat(dataset, n_clip=dataset['n_clip_test'])
        elif dataset == Avenue:
            dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
        write_video_result(dataset, np.array([h, w]), cube_size, clip_idx, model_test, train=not test_set, operation=np.std,
                           frame_gt=dataset['ground_truth'][clip_idx], show_all_score=False,
                           frame_range=frame_ranges[dataset['name']] if dataset in [Belleview, Train] else None,
                           show_clf=dataset in [Belleview, Train], save_plot_exam_only=(task == 11))

    '''======================='''
    ''' Task -8: Write images '''
    '''======================='''
    if task == -8:
        write_example(dataset, np.array([h, w]), cube_size, clip_idx, model_test, operation=np.std, scale_video=not True, wrapall=True)

    '''============================='''
    ''' Task 9: Visualize G filters '''
    '''============================='''
    if task == 9:
        visualize_filters(dataset, cube_size, n_row, n_col, model_idx=model_test)

    '''============================'''
    ''' Task -9: Visualize weights '''
    '''============================'''
    if task == -9:
        get_weights(dataset, np.array([h, w]), cube_size, model_test, np.std, save_as_image=True)

    '''====================================='''
    ''' Task 10: Convert model to visualize '''
    '''====================================='''
    if task == 10:
        convert_model(dataset, cube_size, n_row, n_col, model_idx=model_test)

    print('finished time: %s' % datetime.datetime.now())


if __name__ == '__main__':
    main(sys.argv)