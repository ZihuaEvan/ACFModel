## Requirements (Our Main Enviroment)
+ Python 3.7
+ PyTorch 1.7.1
+ TorchVision 0.8.0
+ gensim
+ lmdbdict
+ yacs

## Install
```bash
python -m pip install -e
pip install -r requirements.txt
```
(**You can find the detailed install steps in [here](https://github.com/ruotianluo/ImageCaptioning.pytorch/)**)

## Data preparing
### Dataset
+ Download MSCOCO dataset from [MSCOCO](https://cocodataset.org/index.htm#download), and unzip it to ./data/
+ Extract features via [feature_extractor](./scripts/feature_extractor.py) using pretrained [ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT) and [Swin](https://github.com/microsoft/Swin-Transformer)

### Labels
Download the dataset infos and the labels from [here](https://drive.google.com/drive/folders/1p2WIh-89fRa61NyqBEuq8U0nyL3jRpFp?usp=share_link) and put it to ./data/ 


## Pretrained models
+ Our checkpoints can be found in [here](https://drive.google.com/drive/folders/1p2WIh-89fRa61NyqBEuq8U0nyL3jRpFp?usp=share_link)
+ Run this to evaluate on Karpathy's test split:
```bash
python eval.py --dump_images 0 --num_images 5000 --model checkpoints/model.pth --infos_path checkpoints/infos.pkl  --language_eval 1 --beam_size 5 --batch_size 50 --gpu 0  --input_label_h5 data/cocoacf_label.h5 --input_json data/cocoacf.json 
```

## Train
+ The training configs can be found in ./acf1.yml
+ Our main code is in [ACFModel](./models/ACFModel.py)
+ Run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --cfg acf1.yml
```
