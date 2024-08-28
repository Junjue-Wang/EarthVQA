import ever as er
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import json
from module import viz
from data.lovedav2 import COLOR_MAP
er.registry.register_all()


def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    seg_metric = er.metric.PixelMetric(8, logdir=self._model_dir, logger=self.logger)
    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = viz.VisualizeSegmm(vis_dir, palette)


    with torch.no_grad():
        for img, ret in tqdm(test_dataloader):
            pred_seg = self.model(img, ret)
            if isinstance(pred_seg, tuple):
                pred_seg = pred_seg[1]
            seg_gt = ret['mask'].cpu().numpy()
            # calculate segmentation accuracy
            pred_seg = pred_seg.argmax(dim=1).cpu().numpy()
            valid_inds = seg_gt != -1
            seg_metric.forward(seg_gt[valid_inds], pred_seg[valid_inds])

            for pred_seg_i, imagen_i in zip(pred_seg, ret['imagen']):
                viz_op(pred_seg_i, imagen_i.replace('jpg', 'png'))


    seg_metric.summary_all()
    torch.cuda.empty_cache()


def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



if __name__ == '__main__':
    seed_torch(42)
    trainer = er.trainer.get_trainer('th_ddp')()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
