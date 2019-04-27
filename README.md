# Semantic Image Synthesis with SPADE - Tensorflow
Simple Tensorflow implementation of ["Semantic Image Synthesis with Spatially-Adaptive Normalization"](https://arxiv.org/abs/1903.07291) (CVPR 2019 Oral)

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── image
           ├── 000001.jpg 
           ├── 000002.png
           └── ...
       ├── segmap
           ├── 000001_segmap.jpg
           ├── 000002_segmap.png
           └── ...
       ├── segmap_test
           ├── a.jpg 
           ├── b.png
           └── ...
       ├── segmap_label.txt (Automatically created) 
       
├── guide.jpg (example for guided image translation task)
```

### Train
```
> python main.py --dataset spade_cityscape --phase train
```

### Random test
```
> python main.py --dataset spade_cityscape --phase random
```

### Guide test
```
python main.py --dataset spade_cityscape --phase guide --guide_img ./guide_img.png
```

## Author
Junho Kim
