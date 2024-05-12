import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os
from tqdm import tqdm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import argparse
import h5py
er.registry.register_all()

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='sfpnr50')
parser.add_argument('--save_dir',  type=str,
                    help='save dir', default='./log/test_seg_features')

args = parser.parse_args()


logger = logging.getLogger(__name__)

er.registry.register_all()


def predict(ckpt_path, config_path='sfpnr50', save_dir='./log/test_seg_features'):
    cfg = import_config(config_path)
    #model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
    model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    log_dir = os.path.dirname(ckpt_path)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            pred, img_feat = model(img.cuda())
            print(img_feat.shape)
            pred = pred.argmax(dim=1).cpu()

            for clsmap, feat_i, imname in zip(pred, img_feat, gt['imagen']):
                clsmap = clsmap.cpu().numpy().astype(np.uint8)
                hdf_path = os.path.join(save_dir, imname.replace('.png', '.hdf5'))
                f = h5py.File(hdf_path, 'w')
                f.create_dataset('feature', data=feat_i.cpu().numpy())
                f.create_dataset('pred_mask', data=clsmap+1)
                f.close()
                
                
                
            torch.cuda.empty_cache()

if __name__ == '__main__':
    predict(args.ckpt_path, args.config_path, args.save_dir)

