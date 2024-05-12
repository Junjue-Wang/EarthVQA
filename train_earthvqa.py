import ever as er
import torch
import numpy as np
import os
from tqdm import tqdm
import random
import json
from utils.metric import VQA_OA_Metric, Count_RMSE_Metric
from data.earthvqa import EarthVQADataset
er.registry.register_all()

def convert2str(indexes, map_dict=EarthVQADataset.QUESTION_VOC):
    if isinstance(indexes, np.int64):
        converted_str = map_dict[indexes]
    else:
        converted_str = ' '.join([map_dict[idx] for idx in indexes if map_dict[idx] != ' ']) + '?'
    return converted_str

COUNT_TYPES = ['Basic Counting', 'Reasoning-based Counting']
COUNT_QUESTIONS = ['What is the area of buildings?', 'What is the area of roads?', 'What is the area of water?', 'What is the area of barren?', 'What is the area of the forest?', 'What is the area of agriculture?', 'What is the area of playgrounds?', 'How many intersections are in this scene?', 'How many eutrophic waters are in this scene?']
def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    metric = VQA_OA_Metric(EarthVQADataset.QUESTION_TYPES, logger=self.logger)
    detail_metric = VQA_OA_Metric(EarthVQADataset.QUESTIONS, logger=self.logger)
    count_metric = Count_RMSE_Metric(COUNT_TYPES, logger=self.logger, uper_cls_idx=17)
    detail_count_metric = Count_RMSE_Metric(COUNT_QUESTIONS, logger=self.logger, uper_cls_idx=17)


    pred_save_path = os.path.join(self._model_dir, f'prediction-{self.checkpoint.global_step}.json')
    att_dir = os.path.join(self._model_dir, 'attention_features')
    os.makedirs(att_dir, exist_ok=True)
    pred_dict = dict()
    with torch.no_grad():
        for img, ret in tqdm(test_dataloader):
            ques, questypes, ans, imagen  = ret['question'], ret['questype'], ret['answer'], ret['imagen']
            preds = self.model(img, ret)
            if isinstance(ques[0], str):
                ques = [q_i+'?' for q_i in ques]
            else:
                ques = [convert2str(q_i, EarthVQADataset.QUESTION_VOC) for q_i in ques]

            ans_idx = preds.argmax(dim=1).cpu().numpy()
            ans = ans.cpu().numpy()
            metric(ans_idx, ans, questypes)

            detail_metric(ans_idx, ans, ques)
            count_metric(ans_idx, ans, questypes)
            detail_count_metric(ans_idx, ans, ques)
            for q_i_str, qt_i, ans_i, imagen_i, gt_i in zip(ques, questypes, ans_idx, imagen, ans):
                qa_list = pred_dict.get(imagen_i, [])
                instace_qa = dict()
                ans_i_str = convert2str(ans_i, EarthVQADataset.ANSWER_VOC)
                gt_i_str = convert2str(gt_i, EarthVQADataset.ANSWER_VOC)
                instace_qa['Type'] = qt_i
                instace_qa['Question'] = q_i_str
                instace_qa['Answer'] = ans_i_str
                instace_qa['Correct'] = str(ans_i == gt_i)
                instace_qa['GT_Answer'] = gt_i_str
                qa_list.append(instace_qa)
                pred_dict[imagen_i] = qa_list


    with open(pred_save_path,'w',encoding='utf-8') as f:
        f.write(json.dumps(pred_dict, ensure_ascii=False, indent=1))

    metric.summary()
    detail_metric.summary()
    count_metric.summary()
    detail_count_metric.summary()
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True



if __name__ == '__main__':
    seed_torch(42)
    trainer = er.trainer.get_trainer('th_ddp')()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
