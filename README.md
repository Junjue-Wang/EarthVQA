<h2 align="center">EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering</h2>

<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>, <a href="http://zhuozheng.top/">Zhuo Zheng</a>, Zihang Chen, Ailong Ma, and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a></h5>

[[`Paper`](https://www.researchgate.net/publication/376519677_EarthVQA_Towards_Queryable_Earth_via_Relational_Reasoning-Based_Remote_Sensing_Visual_Question_Answering)],
[[`Video`](https://s3.amazonaws.com/pf-user-files-01/u-59356/uploads/2024-01-08/2q83o0t/EarthVQA-video.mp4)],
[[`Dataset`](http://um7780bis3qkjojc.mikecrm.com/hsnbBLR)],
[[`Leaderboard-SEG`](https://www.codabench.org/competitions/2921)],
[[`Leaderboard-VQA`](https://www.codabench.org/competitions/2922)]

<div align="center">
  <img src="https://github.com/Junjue-Wang/resources/blob/main/EarthVQA/Dataset-vis.png?raw=true">
</div>

## News
- 2024/05/12, Code and Pre-trained weights have been updated.

- 2024/05/11, EarthVQA dataset has been released.




#### Requirements:
- pytorch >= 1.1.0
- python >=3.6


### Install Ever + Segmentation Models PyTorch
```bash
pip install ever-beta
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
### Data preparation
- Download EarthVQA dataset and pre-trained weights
- Construct the data as follows:
```none
EarthVQA
├── Train
│   ├── images_png
│   ├── masks_png_v2
├── Val
│   ├── images_png
│   ├── masks_png_v2
├── Test
│   ├── images_png
├── Train_QA.json
├── Val_QA.json
├── Test_QA.json
log
├── sfpnr50.pth
├── soba.pth
```
### Test

```bash
# 1. generate semantic masks use the pre-trained SFPN weight
sh ./scripts/generate_segfeats.sh
# 2. generate answers use the pre-trained SOBA weight
sh ./scripts/predict_soba.sh
```
### Train
```bash
# 1 train a segmentation model
sh ./scripts/train_sfpnr50.sh
# 2 generate segmentation features and pse-masks
sh ./scripts/generate_segfeats.sh
# 3 train SOBA
sh ./scripts/train_soba.sh
```

## Citation
If you use EarthVQA in your research, please cite our following papers.
```text
    @article{wang2024earthvqa, 
        title={EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering},
        url={https://ojs.aaai.org/index.php/AAAI/article/view/28357}, 
        DOI={10.1609/ai.v38i6.28357}, 
        author={Junjue Wang and Zhuo Zheng and Zihang Chen and Ailong Ai and Yanfei Zhong}, 
        year={2024}, 
        month={Mar.},
        volume={38},
        pages={5481-5489}}
    @article{earthvqanet,
        title = {EarthVQANet: Multi-task visual question answering for remote sensing image understanding},
        journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
        volume = {212},
        pages = {422-439},
        year = {2024},
        issn = {0924-2716},
        doi = {https://doi.org/10.1016/j.isprsjprs.2024.05.001},
        url = {https://www.sciencedirect.com/science/article/pii/S0924271624001990},
        author = {Junjue Wang and Ailong Ma and Zihang Chen and Zhuo Zheng and Yuting Wan and Liangpei Zhang and Yanfei Zhong},
    }
```


## Dataset and Contest
The EarthVQA dataset is released at [<b>Google Drive</b>](http://um7780bis3qkjojc.mikecrm.com/hsnbBLR)
and [<b>Baidu Drive</b>](http://um7780bis3qkjojc.mikecrm.com/PnyrDF3)



You can develop your models on Train and Validation sets.

Semantic Category labels: background – 1, building – 2, road – 3,
                 water – 4, barren – 5,forest – 6, agriculture – 7, playground - 8. And the no-data regions were assigned 0
                 which should be ignored. The provided data loader will help you construct your pipeline.  
                 

Submit your test results on [<b>EarthVQA Semantic Segmentation Challenge</b>](https://www.codabench.org/competitions/2921), 
[<b>EarthVQA Visual Question Answering Challenge</b>](https://www.codabench.org/competitions/2922).
You will get your Test scores smoothly.

Feel free to design your own models, and we are looking forward to your exciting results!


## License
The owners of the data and of the copyright on the data are [RSIDEA](http://rsidea.whu.edu.cn/), Wuhan University.
Use of the Google Earth images must respect the ["Google Earth" terms of use](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).
All images and their associated annotations in EarthVQA can be used for academic purposes only,
<font color="red"><b> but any commercial use is prohibited.</b></font>

<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">
<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
