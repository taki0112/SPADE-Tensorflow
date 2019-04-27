# Semantic Image Synthesis with SPADE - Tensorflow
<div align="center">
  <img src="./assets/teaser.png">
</div>

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
> python main.py --dataset spade_cityscape --phase guide --guide_img ./guide_img.png
```

## Architecture
*Generator* | *Image Encoder* | *Discriminator* | *All-in-one* |
:---: | :---: | :---: | :---: |
<img src = './assets/generator.png' width = '400px' height = '400px'> | <img src = './assets/image_encoder.png' width = '400px' height = '400px'> | <img src = './assets/discriminator.png' width = '350px' height = '350px'> | <img src = './assets/architecture.png' width = '400px' height = '400px'> |

### SPADE architecture
*SPADE* | *SPADE Residual Block* | 
:---: | :---: |
<img src = './assets/spade.png' width = '1000px' height = '400px'> | <img src = './assets/spade_resblock.png' width = '420px' height = '400px'> |

## Results
Soon

## Author
Junho Kim
