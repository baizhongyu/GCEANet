# GCEANet
The source code of the paper: Side-Scan Sonar Image Recognition with Zero-Shot and Style Transfer


## Requirements  
We recommend the following configurations:  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1


## Model Training  
- Download the content dataset: [MS-COCO](https://cocodataset.org/#download).
- Download the style dataset: [WikiArt](https://www.kaggle.com/c/painter-by-numbers).
- Download the pre-trained [VGG-19](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing) model.
- Set your available GPU ID in Line94 of the file "train.py".
- Run the following command:
```
python train.py --content_dir /data/train2014 --style_dir /data/WikiArt/train
```
## Model Testing
- All optical images of the sonar style transfer test can be downloaded: [optical-content](https://1drv.ms/u/s!AhLjganHO9NJgt0prZHFLV8MTjmnPQ?e=wDYnsQ).
- The sonar images as style images used for pseudo-SSS image synthesis are at *./sonar_test.
- Run the following command:
```
python Eval.py --content input/content/1.jpg --style input/style/1.jpg
```
Our model will be published as soon as possible.

 ## Acknowledgments
The code in this repository is based on [SANet](https://github.com/GlebBrykin/SANET). Thanks for both their paper and code.

