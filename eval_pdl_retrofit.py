import time
import paddle
import paddle.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
import xlwt
from src.utils.timer import Timer
import src.utils_pdl.evaluation_metric as metric

from data.data_loader_pdl import GMDataset, get_dataloader
# TODO: Support for dataparallel
from typing import List
from paddle.io import DataLoader
from pygmtools.benchmark import Benchmark

from src.utils.config import cfg


def eval_model(
        model: nn.Layer,
        classes: List[str],
        bm,
        last_epoch=True,
        verbose=False,
        xls_sheet=None):
    print('Start Evaluation.')
    start_time = time.time()

    # TODO: automatic detect model device
    device = 'gpu:0'
    paddle.set_device(device)

    was_training = model.training
    model.eval()

    dataloaders: List[DataLoader] = []

    for cls in classes:
        image_dataset = Benchmark(
            name=cfg.DATASET_FULL_NAME,
            bm=bm,
            problem=cfg.PROBLEM.TYPE,
            length=cfg.EVAL.SAMPLES,
            cls=cls,
            using_all_graphs=cfg.PROBLEM.TEST_ALL_GRAPHS)
        paddle.seed(cfg.RANDOM_SEED)

        dataloader = get_dataloader(image_dataset, shuffle=True)
        dataloaders.append(dataloader)

    recalls = []
    precisions = []
    f1s = []
    coverages = []
    pred_time = []
    objs = paddle.to_tensor(np.zeros(len(classes)))
    cluster_acc = []
    cluster_purity = []
    cluster_ri = []

    timer = Timer()

    prediction = []

    for i, cls in enumerate(classes):
        if verbose:
            print(
                'Evaluating class '
                + f'{cls}: {i}/{len(classes)}')

        running_start = time.time()
        iter_num = 0

        pred_time_list = []
        obj_total_num = paddle.zeros(1)
        cluster_acc_list = []
        cluster_purity_list = []
        cluster_ri_list = []
        prediction_cls = []

        for inputs in dataloaders[i]:
            if iter_num >= cfg.EVAL.SAMPLE / inputs['batch_size']:
                break

            batch_size = inputs['batch_size']

            iter_num += 1

            with paddle.set_grad_enabled(False):
                timer.tick()
                outputs = model.forward(inputs)
                pred_time_list.append(
                    paddle.full((batch_size,), timer.toc() / batch_size))

            # evaluate matching acc
            if cfg.PROBLEM.TYPE != '2GM':
                raise NotImplementedError(
                    'Only 2GM problems are supported for now.')
            assert 'perm_mat' in outputs

            for b in range(outputs['perm_mat'].shape[0]):
                perm_mat = outputs['perm_mat'][
                    b, :outputs['ns'][0][b], :outputs['ns'][1][b]]
                perm_mat = perm_mat.numpy()
                eval_dict = dict()
                id_pair = inputs['id_list'][0][b], inputs['id_list'][1][b]
                eval_dict['ids'] = id_pair
                eval_dict['cls'] = cls
                eval_dict['perm_mat'] = perm_mat
                prediction.append(eval_dict)
                prediction_cls.append(eval_dict)

            if 'aff_mat' in outputs:
                pred_obj_score = metric.objective_score(
                    outputs['perm_mat'], outputs['aff_mat'])
                gt_obj_score = metric.objective_score(
                    outputs['gt_perm_mat'], outputs['aff_mat'])
                objs[i] += paddle.sum(pred_obj_score / gt_obj_score)
                obj_total_num += batch_size

            if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
                running_speed = cfg.STATISTIC_STEP * batch_size /\
                    (time.time() - running_start)
                print(
                    'Class {:<8} Iter {:<4} {:>4.2f}sample/s'
                    .format(cls, iter_num, running_speed))
                running_start = time.time()

        objs[i] = objs[i] / obj_total_num
        pred_time.append(paddle.concat(cluster_acc_list))

        if verbose:
            bm.eval_cls(prediction_cls, cls, verbose=verbose)
            print(
                f'Class {cls} norm obj score: {objs[i]:.4f}')
            print(
                f'Class {cls} pred time: {metric.format_metric(pred_time[i])}')

    result = bm.eval(prediction, classes, verbose=True)
    for cls in classes:
        precision = result[cls]['precision']
        recall = result[cls]['recall']
        f1 = result[cls]['f1']
        coverage = result[cls]['coverage']

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        coverages.append(coverage)

    time_elapsed = time.time() - start_time
    minutes = time_elapsed // 60
    seconds = time_elapsed % 60
    print(f'Evaluation completed in {minutes:.0f}m {seconds:.0f}s')

    if was_training:
        model.train()

    # TODO: add xlsx table output support
    if not paddle.any(paddle.isnan(objs)):
        print('Normalized Objective Score')
        for idx, (cls, cls_obj) in enumerate(zip(classes, objs)):
            print(f'{cls} = {cls_obj:.4f}')
        print(f'Average Objective Score = {paddle.mean(objs):.4f}')
