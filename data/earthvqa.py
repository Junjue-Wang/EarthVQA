import warnings
import numpy as np
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import os
import logging
from skimage.io import imread
import ever as er
from ever.interface import ConfigurableMixin
from collections import OrderedDict
from data import distributed
import json
import h5py


logger = logging.getLogger(__name__)


COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
    Playground=(165,0,165),
    Pond=(0,185,246),
)




class EarthVQADataset(Dataset):
    QUESTIONS = ['Are there any villages in this scene?', 'Is there a commercial area near the residential area?', 'Are there any playgrounds in this scene?', 'Is there any commercial land in this scene?', 'Is there any forest in this scene?', 'Is there any agriculture in this scene?', 'What are the types of residential buildings?', 'Are there any urban villages in this scene?', 'What are the needs for the renovation of villages?', 'Is there any barren in this scene?', 'Whether greening need to be supplemented in residential areas?', 'Is there any woodland in this scene?', 'What are the land use types in this scene?', 'What are the needs for the renovation of residents?', 'Are there any buildings in this scene?', 'Is there any agricultural land in this scene?', 'What is the area of roads?', 'What is the area of playgrounds?', 'Is it a rural or urban scene?', 'What is the area of barren?', 'Are there any bridges in this scene?', 'Are there any eutrophic waters in this scene?', 'Are there any viaducts in this scene?', 'What is the area of water?', 'Are there any roads in this scene?', 'Is there any residential land in this scene?', 'How many eutrophic waters are in this scene?', 'Is there any industrial land in this scene?', 'Is there any park land in this scene?', 'Is there any uncultivated agricultural land in this scene?', 'Is there a school near the residential area?', 'Are there any large driveways (more than four lanes)?', 'What are the comprehensive traffic situations in this scene?', 'What is the area of buildings?', 'Is there any construction land in this scene?', 'What are the water types in this scene?', 'Are there any viaducts near the residential area?', 'Is there a construction area near the residential area?', 'Is there a park near the residential area?', 'What are the road materials around the village?', 'Are there any intersections in this scene?', 'What are the road types around the residential area?', 'What are the water situations around the agricultural land?', 'What is the situation of barren land?', 'What is the area of the forest?', 'Are there any intersections near the school?', 'Is there any water in this scene?', 'Is there any educational land in this scene?', 'How many intersections are in this scene?', 'Are there any greenhouses in this scene?', 'What is the area of agriculture?']
    QUESTION_VOC = [' ', 'materials', 'are', 'driveways', 'How', 'construction', 'industrial', 'be', 'area', 'barren', 'agricultural', 'road', 'educational', 'intersections', 'village', 'greenhouses', 'any', 'many', 'bridges', 'areas', 'scene', 'there', 'buildings', 'uncultivated', 'traffic', 'Is', 'residential', 'forest', 'woodland', 'supplemented', 'near', 'than', 'Are', 'eutrophic', 'this', 'playgrounds', 'situations', 'commercial', 'urban', 'land', 'school', 'residents', '(more', 'it', 'rural', 'viaducts', 'is', 'types', 'to', 'roads', 'the', 'for', 'greening', 'of', 'four', 'a', 'park', 'comprehensive', 'agriculture', 'in', 'What', 'villages', 'needs', 'around', 'water', 'situation', 'use', 'waters', 'or', 'large', 'need', 'lanes)', 'renovation', 'Whether']
    QUESTION_TYPES = ['Basic Judging', 'Reasoning-based Judging',  'Basic Counting', 'Reasoning-based Counting', 'Object Situation Analysis', 'Comprehensive Analysis']
    ANSWER_VOC = [0, 1, 2, 3, 4, 5, 6, '0%-10%', '10%-20%', '20%-30%','30%-40%','40%-50%','50%-60%','60%-70%','70%-80%','80%-90%', '90%-100%',  'The roads need to be improved, and waters need to be cleaned up',  'This is an important traffic area with 3 intersections', 'There are residential, educational, park, and agricultural areas', 'Developing', 'There are railways',  'This is a very important traffic area with 1 intersection, several viaducts, and several bridges', 'There are cement roads', 'There are educational, construction, and agricultural areas', 'Underdeveloped', 'There are unsurfaced roads, and cement roads', 'There are residential, commercial, park, and agricultural areas', 'There are commercial areas', 'This is a very important traffic area with 2 intersections, and several viaducts', 'There are commercial, construction, and park areas', 'There are residential, commercial, park, industrial, and agricultural areas', 'There are commercial, and construction areas', 'This is not an important traffic area', 'This is a very important traffic area with 2 intersections, and several bridges', 'There are unsurfaced roads, and railways', 'There are woodland, industrial, and agricultural areas', 'There are park areas', 'There are construction, park, and agricultural areas', 'There are residential, and industrial areas', 'There are residential, and construction areas', 'There is no water', 'There are residential, construction, and park areas', 'There are commercial buildings', 'There are agricultural areas', 'There are educational areas', 'There are residential, and commercial areas', 'There are commercial, educational, park, and industrial areas', 'There are clean waters near the agriculture land', 'There are ponds', 'There are residential, commercial, park, and industrial areas', 'There are educational, park, industrial, and agricultural areas', 'There are unsurfaced roads, cement roads, railways, and asphalt roads', 'There are one-way lanes, and railways', 'There are residential, commercial, educational, park, and industrial areas', 'There are no water area', 'There are railways, and asphalt roads', 'There are construction areas', 'The urban villages need attention', 'There are unsurfaced roads, railways, and asphalt roads', 'There are residential, and agricultural areas', 'There are residential, commercial, and agricultural areas', 'No', 'This is a very important traffic area with 1 intersection, and several viaducts', 'The greening needs to be supplemented', 'There are residential, commercial, educational, and construction areas', 'This is an important traffic area with several bridges', 'There are residential, commercial, educational, and industrial areas', 'There are woodland areas', 'There are residential, commercial, and construction areas', 'Rural', 'There are residential, construction, park, industrial, and agricultural areas', 'There are residential, woodland, industrial, and agricultural areas', 'This is an important traffic area with 4 intersections', 'There are private buildings', 'There are woodland, and agricultural areas', 'There are residential, commercial, construction, and park areas',  'There are rivers and ponds', 'There are residential, construction, and agricultural areas', 'There are residential, and educational areas', 'There are commercial, and educational areas', 'There are polluted waters near the agriculture land', 'There are one-way lanes, wide lanes, and railways', 'There are one-way lanes, and wide lanes', 'Urban', 'There are residential, commercial, and educational areas', 'There are commercial, and park areas', 'There are unsurfaced roads, cement roads, and asphalt roads', 'There are commercial buildings, and private buildings',  'This is an important traffic area with 1 intersection', 'There are commercial, industrial, and agricultural areas', 'There are residential, commercial, construction, park, and industrial areas', 'There are asphalt roads', 'There are residential, commercial, and park areas', 'There are no agricultural land',  'There are commercial, construction, park, and agricultural areas', 'There are residential, educational, and construction areas', 'There are commercial, construction, and industrial areas', 'There are residential, commercial, construction, and industrial areas', 'There are park, and industrial areas', 'There are commercial, and agricultural areas', 'There are residential, educational, construction, and park areas', 'No obvious land use types', 'There are construction, park, and industrial areas', 'There are residential, educational, park, and industrial areas', 'There are commercial, park, and industrial areas', 'This is an important traffic area with several viaducts', 'This is a very important traffic area with 1 intersection, and several bridges', 'There are residential, park, and agricultural areas', 'There are residential, commercial, construction, and agricultural areas', 'There are residential, commercial, educational, construction, park, and agricultural areas', 'There are wide lanes, and railways', 'There are residential, park, and industrial areas', 'There are residential, industrial, and agricultural areas', 'There are construction, and park areas', 'There are residential, commercial, construction, park, industrial, and agricultural areas', 'There are residential, park, industrial, and agricultural areas', 'There are residential areas', 'There are residential, commercial, educational, park, and agricultural areas', 'There are residential, commercial, industrial, and agricultural areas', 'There are residential, commercial, educational, and park areas', 'There are construction, and agricultural areas', 'There are no water nor agricultural land', 'The waters need to be cleaned up', 'There are park, and agricultural areas', 'There are rivers', 'This is a very important traffic area with 3 intersections, and several viaducts', 'This is an important traffic area with 2 intersections', 'There are industrial areas', 'There are unsurfaced roads, and asphalt roads', 'This is a very important traffic area with 2 intersections, several viaducts, and several bridges', 'There are commercial, park, and agricultural areas', 'There are one-way lanes', 'There are residential, educational, construction, and agricultural areas', 'There are no roads', 'There are residential, construction, park, and agricultural areas', 'There are residential, and park areas', 'There are commercial, construction, and agricultural areas', 'There are cement roads, and asphalt roads',  'There are residential, educational, and agricultural areas', 'There are commercial, and industrial areas', 'There are park, industrial, and agricultural areas', 'This is a very important traffic area with several viaducts, and several bridges', 'There are educational, construction, and park areas', 'There are residential, woodland, and agricultural areas', 'There are residential, and woodland areas', 'There are unsurfaced roads, cement roads, and railways', 'There are educational, park, and agricultural areas', 'There are residential, educational, and park areas', 'There are commercial, educational, and park areas', 'There are wide lanes', 'There are cement roads, and railways', 'There are no residential buildings', 'There are commercial, park, industrial, and agricultural areas', 'There are residential, commercial, and industrial areas', 'The greening needs to be supplemented and urban villages need attention', 'There is no barren land', 'There are educational, and agricultural areas', 'The roads need to be improved', 'Yes', 'There are unsurfaced roads', 'There are residential, commercial, construction, park, and agricultural areas', 'There are residential, construction, and industrial areas', 'There are cement roads, railways, and asphalt roads', 'There are educational, and park areas', 'There are no needs']


    def __init__(self, image_dir='', mask_dir='', qa_path='', image_feat_dir='', encoding=True, common_transforms=None, image_transforms=None, vqa_transforms=None,
                 image_load=True, mask_load=True, feat_load=False, pred_mask_load=False, feat_shuffle=False):
        self.item_list = []
        self.question_voc = []
        self.ans_voc = []
        self.encoding = encoding
        self.image_transforms = image_transforms
        self.vqa_transforms = vqa_transforms
        self.common_transforms = common_transforms
        self.feat_shuffle = feat_shuffle
        self.pred_mask_load = pred_mask_load
        if isinstance(qa_path, str):
            qa_path = [qa_path]
        if isinstance(image_dir, str):
            image_dir = [image_dir] * len(qa_path) if len(qa_path)>1 else [image_dir]
        if isinstance(mask_dir, str):
            mask_dir = [mask_dir] * len(qa_path) if len(qa_path)>1 else [mask_dir]
        if isinstance(image_feat_dir, str):
            image_feat_dir = [image_feat_dir] * len(qa_path)
        for image_d, mask_d, qa_p, image_f in zip(image_dir, mask_dir, qa_path, image_feat_dir):
            self.generate_pairs(image_d, mask_d, qa_p, image_f)

        self.question_voc = list(set(self.question_voc))
        self.ans_voc = list(set(self.ans_voc))
        self.image_load = image_load
        self.mask_load = mask_load
        self.feat_load = feat_load
        print(f'The {len(self.item_list)} QAs are loaded!')
        print(f'The question voc has {len(self.question_voc)} words!')
        print(f'The answer voc has {len(self.ans_voc)} words!')

    def generate_pairs(self, image_dir, mask_dir, qa_path, image_feat_dir=None):
        qas_dict = json.load(open(qa_path, 'r'))
        for imagen, qas_list in qas_dict.items():
            for qa_dict in qas_list:
                questype, ques, ans = qa_dict.values()
                # Question ans count
                words = ques.strip('?').split(' ')
                self.question_voc += words
                self.ans_voc.append(ans)
                self.item_list.append(dict(image_path=os.path.join(image_dir, imagen),
                                           mask_path=os.path.join(mask_dir, imagen),
                                           feat_path=os.path.join(image_feat_dir, imagen.replace('png', 'hdf5')),
                                           questype=questype,
                                           ques=words,
                                           ans=ans,
                                           ))

    def __getitem__(self, idx):
        image_path, mask_path, feat_path, questype, ques, ans = self.item_list[idx].values()
        mask = []
        vqa_image = []
        if self.image_load:
            vqa_image = imread(image_path)
            
        if self.mask_load:
            mask = imread(mask_path).astype(np.int64) - 1

        if self.feat_load:
            feat = h5py.File(feat_path,'r')
            vqa_image = np.array(feat['feature']).transpose([1, 2, 0])
            if self.pred_mask_load:
                mask = np.array(feat['pred_mask']).astype(np.int64) - 1

        
        if self.vqa_transforms is not None:
            if self.pred_mask_load:
                blob = self.vqa_transforms(image=vqa_image, mask=mask)
                image, mask = blob['image'], blob['mask']
            else:
                image = self.vqa_transforms(image=vqa_image)['image']
                
        if self.feat_load:
            vqa_image = vqa_image.transpose([2, 0, 1])
            
        question_len = len(ques)
        if self.encoding:
            encoded_ques = np.zeros(len(self.QUESTION_VOC)).astype(np.int64)
            for i, word in enumerate(ques):
                encoded_ques[i] = EarthVQADataset.QUESTION_VOC.index(word)
            ques = encoded_ques
        else:
            ques = ' '.join(ques)
        
        if ans != '':
            ans = EarthVQADataset.ANSWER_VOC.index(ans)
       
        return vqa_image, dict(imagen=os.path.basename(image_path), vqaimagen=os.path.basename(image_path),
         seg_mask=mask, question=ques, questype=questype, question_len=question_len, answer=ans)

    def __len__(self):
        return len(self.item_list)


@er.registry.DATALOADER.register()
class EarthVQALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = EarthVQADataset(self.config.image_dir, self.config.mask_dir,
                                 self.config.qa_path, self.config.image_feat_dir,
                                 self.config.encoding, self.config.common_transforms, self.config.image_transforms, self.config.vqa_transforms,
                                 self.config.image_load, self.config.mask_load, self.config.feat_load, self.config.pred_mask_load,
                                 self.config.feat_shuffle
                                 )
        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(
            dataset)

        super(EarthVQALoader, self).__init__(dataset,
                                            self.config.batch_size,
                                            sampler=sampler,
                                            num_workers=self.config.num_workers,
                                            pin_memory=True,
                                            drop_last=True if self.config.training else False
                                            )

    def set_default_config(self):
        self.config.update(dict(
            encoding=True,
            image_load=True,
            mask_load=True,
            feat_load=False,
            feat_shuffle=False,
            pred_mask_load=False,
            vqa_transforms=None,
            common_transforms=None,
            image_dir='',
            mask_dir='',
            qa_path='',
            image_feat_dir='',
            image_transforms=None,
            batch_size=3,
            num_workers=0,
            training=True
        ))
