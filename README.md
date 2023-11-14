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
- Put your trained model to *./model/* folder.
- Set your available GPU ID in Line 92 of the file "train.py".
- Run the following command:
```
python train.py --content_dir /data/train2014 --style_dir /data/WikiArt/train
```
## pseudo-SSS image synthesis
- All optical images of the sonar style transfer test can be downloaded: [optical-content](https://1drv.ms/u/s!AhLjganHO9NJgt0prZHFLV8MTjmnPQ?e=wDYnsQ).
- Put the optical images to *./Optic_test/* folder.
- The sonar images as style images used for pseudo-SSS image synthesis are at *./sonar_test.
- Run the following command:
```
python batch_convert.py
```
Our model will be published as soon as possible.


## Real SSS image synthesis
- The real sonar images can be downloaded: [real_dataset](https://github.com/guizilaile23/ZSL-SSS) and [KLSG](https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset).


 ## Acknowledgments
The code in this repository is based on [Li et al.](https://github.com/guizilaile23/ZSL-SSS) and [SANet](https://github.com/GlebBrykin/SANET). Thanks for both their paper and code.

