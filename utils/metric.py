import numpy as np
import logging
import prettytable as pt
import math

logging.basicConfig(level=logging.INFO)


class VQA_OA_Metric(object):
    def __init__(self, ques_classes:list, logger=None):
        self.ques_classes = ques_classes
        self.logger = logger
        self.cls_true_num = [0 for _ in range(len(self.ques_classes))]
        self.cls_total_num = [0 for _ in range(len(self.ques_classes))]

    def __call__(self, pred:np.array, gt:np.array, questype):
        matched = (pred == gt).astype(np.uint8)
        
        for ques_t, m_i in zip(questype, matched):
            cls_idx = self.ques_classes.index(ques_t)
            self.cls_total_num[cls_idx] += 1
            self.cls_true_num[cls_idx] += int(m_i)

    def summary(self):
        tb = pt.PrettyTable()
        tb.field_names = ['Classes', 'Acc']
        for idx in range(len(self.cls_true_num)):
            tb.add_row([self.ques_classes[idx], self.cls_true_num[idx] / self.cls_total_num[idx]])

        tb.add_row(['OA', sum(self.cls_true_num) / sum(self.cls_total_num)])

        if self.logger is None:
            print('\n' + tb.get_string())
        else:
            self.logger.info('\n' + tb.get_string())

def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None



class Count_RMSE_Metric(object):
    def __init__(self, ques_classes:list, logger=None, uper_cls_idx=26, many_scale_factor=10/7):
        self.ques_classes = ques_classes
        self.logger = logger
        self.uper_cls_idx = uper_cls_idx
        self.gts = [[] for _ in range(len(self.ques_classes))]
        self.preds = [[] for _ in range(len(self.ques_classes))]
        self.many_scale_factor = many_scale_factor
        

    def __call__(self, pred:np.array, gt:np.array, questype):
        masked = gt < self.uper_cls_idx
        pred, gt, questype = pred[masked], gt[masked], np.array(questype)[masked]
        for idx, ques_t in enumerate(questype):
            cls_idx = self.ques_classes.index(ques_t)
            self.gts[cls_idx].append(gt[idx])
            self.preds[cls_idx].append(pred[idx])

    def summary(self):
        tb = pt.PrettyTable()
        tb.field_names = ['Classes', '↓RMSE']
        for idx in range(len(self.ques_classes)):
            ques = self.ques_classes[idx]
            if 'Reasoning-based' in ques:
                self.gts[idx] = [v * self.many_scale_factor for v in self.gts[idx]]
                self.preds[idx] = [v * self.many_scale_factor for v in self.preds[idx]]

            rmse = get_rmse(self.gts[idx], self.preds[idx])

            tb.add_row([ques, rmse])

        overall_gt, overall_pred = sum(self.gts, []), sum(self.preds, [])
        tb.add_row(['Overall_RSME', get_rmse(overall_gt, overall_pred)])

        if self.logger is None:
            print('\n' + tb.get_string())
        else:
            self.logger.info('\n' + tb.get_string())

