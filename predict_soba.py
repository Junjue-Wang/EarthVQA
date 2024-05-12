import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os, json
from tqdm import tqdm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import argparse
import h5py
from data.earthvqa import EarthVQADataset
er.registry.register_all()

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--pred_save_path',  type=str,
                    help='pred_save_path', default='./test.json/')

args = parser.parse_args()


logger = logging.getLogger(__name__)

er.registry.register_all()


def convert2str(indexes, map_dict=EarthVQADataset.QUESTION_VOC):
    if isinstance(indexes, np.int64):
        converted_str = map_dict[indexes]
    else:
        converted_str = ' '.join([map_dict[idx] for idx in indexes if map_dict[idx] != ' ']) + '?'
    return converted_str

def predict(ckpt_path, config_path='soba', pred_save_path='./'):
    cfg = import_config(config_path)
    #model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
    model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    test_dataloader = make_dataloader(cfg['data']['test'])
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()
    
    pred_dict = dict()
    with torch.no_grad():
        for img, ret in tqdm(test_dataloader):
            ques, questypes, imagen  = ret['question'], ret['questype'], ret['imagen']
            preds = model(img, ret)
            if isinstance(ques[0], str):
                ques = [q_i+'?' for q_i in ques]
            else:
                ques = [convert2str(q_i, EarthVQADataset.QUESTION_VOC) for q_i in ques]

            ans_idx = preds.argmax(dim=1).cpu().numpy()
            
            for q_i_str, qt_i, ans_i, imagen_i in zip(ques, questypes, ans_idx, imagen):
                qa_list = pred_dict.get(imagen_i, [])
                instace_qa = dict()
                ans_i_str = convert2str(ans_i, EarthVQADataset.ANSWER_VOC)
                instace_qa['Type'] = qt_i
                instace_qa['Question'] = q_i_str
                instace_qa['Answer'] = ans_i_str
                qa_list.append(instace_qa)
                pred_dict[imagen_i] = qa_list


    with open(pred_save_path,'w',encoding='utf-8') as f:
        f.write(json.dumps(pred_dict, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    predict(args.ckpt_path, args.config_path, args.pred_save_path)

