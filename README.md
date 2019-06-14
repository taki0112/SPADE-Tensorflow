# Semantic Image Synthesis with SPADE (GauGAN) - Tensorflow

Simple Tensorflow implementation of "Semantic Image Synthesis with Spatially-Adaptive Normalization" (CVPR 2019 Oral)

<div align="center">
  <img src="https://nvlabs.github.io/SPADE/images/treepond.gif">
  <img src="https://nvlabs.github.io/SPADE/images/ocean.gif">
</div>

### [Project page](https://nvlabs.github.io/SPADE/) | [Paper](https://arxiv.org/abs/1903.07291) | [Pytorch code](https://github.com/NVlabs/SPADE)

## Requirements
* scipy == 1.2.0
  * The latest version is not available. `imsave` is deprecated.
* tqdm
* numpy
* pillow
* opencv-python
* tensorflow-gpu
* keras
 
## Preparation
* **YOUR DATASET**
  * Image
  * Segmentation map
    * Don't worry. I do one-hot encoding of segmentation map automatically (whether color or gray)
* **CelebAMask-HQ**
  * Download [**CelebAMask-HQ**](https://drive.google.com/file/d/1ybGz_7uMOjMpAySIA5j3nJSKSwyhSQO0/view?usp=sharing) (pre-processed by ***Junho Kim***)
    * The original was taken from [here](https://github.com/switchablenorms/CelebAMask-HQ)
    
### Pretrained model
* Download [**checkpoint**](https://drive.google.com/file/d/1UIj7eRJeNWrDS-3odyaoLhcqk0tNcEez/view?usp=sharing)


## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── image
           ├── 000001.jpg 
           ├── 000002.png
           └── ...
       ├── segmap
           ├── 000001.jpg
           ├── 000002.png
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
> python main.py --dataset spade_celebA --img_ch 3 --segmap_ch 3 --phase train 
```

### Random test
```
> python main.py --dataset spade_celebA --segmap_ch 3 --phase random
```

### Guide test
```
> python main.py --dataset spade_celebA --img_ch 3 --segmap_ch 3 --phase guide --guide_img ./guide_img.png
```
  
## Our Results 

### Loss grpah
<div align="center">
  <img src="./assets/loss.png">
</div>

### CityScape
<div align="center">
  <img src="./assets/result_img/cityscape_hinge.png">
</div>

### CelebA-HQ (Style Manipulation)
<div align="center">
  <img src="./assets/result_img/women_hinge.png">
</div>

---

<div align="center">
  <img src="./assets/result_img/men_hinge.png">
</div>


### CelebA-HQ (Random Manipulation)
<div align="center">
  <img src="./assets/result_img/women_random_hinge.png">
</div>

---

<div align="center">
  <img src="./assets/result_img/men_random_hinge.png">
</div>

## How about the Least-Square loss ?
### CelebA-HQ (Style Manipulation)
<div align="center">
  <img src="./assets/result_img/women_lsgan.png">
</div>

---

<div align="center">
  <img src="./assets/result_img/men_lsgan.png">
</div>


### CelebA-HQ (Random Manipulation)
<div align="center">
  <img src="./assets/result_img/women_random_lsgan.png">
</div>

---

<div align="center">
  <img src="./assets/result_img/men_random_lsgan.png">
</div>

## Architecture
*Generator* | *Image Encoder* | *Discriminator* | *All-in-one* |
:---: | :---: | :---: | :---: |
<img src = './assets/generator.png' width = '400px' height = '400px'> | <img src = './assets/image_encoder.png' width = '400px' height = '400px'> | <img src = './assets/discriminator.png' width = '350px' height = '350px'> | <img src = './assets/architecture.png' width = '400px' height = '400px'> |

### SPADE architecture
*SPADE* | *SPADE Residual Block* | 
:---: | :---: |
<img src = './assets/spade.png' width = '850px' height = '300px'> | <img src = './assets/spade_resblock.png' width = '500px' height = '400px'> |

## Author
[Junho Kim](http://bit.ly/jhkim_ai)
