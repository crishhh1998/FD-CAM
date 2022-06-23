# FD-CAM: Improving Faithfulness and Discriminability of Visual Explanation for CNNs (ICPR 2022)

[FD-CAM](https://arxiv.org/abs/2206.08792) is a novel CAM weighting scheme which **combines the gradient and score based weights** to improve the faithfulness and discriminability of visual explanation for CNNs. 

## Cite Us

**If you find this repository helpful in your work or research, we would greatly appreciate citations to the following paper:**

```
@inproceedings{FDCAM,
  title={FD-CAM: Improving Faithfulness and Discriminability of Visual Explanation for CNNs},
  author={Hui Li, Zihao Li, Rui Ma, Tieru Wu},
  booktitle= International Conference on Pattern Recognition,
  year={2022}
}
```

## Requirements

### **Main dependencies:**

- torch==1.8.1
- torchvision==0.9.1
- Pillow==8.4.0
- matplotlib==3.2.2
- numpy==1.19.2
- grad-cam==1.3.2

To install requirements, run:

```
pip install -r requirements.txt
```

## Usage

### dataset

We use ILSVRC2015 val set and VOC2007 val set as dataset.

### model

We get pretrained model VGG16 as the model to be explained from PyTorch model zoo.

```
model = torchvision.models.vgg16(pretrained=True).eval()
```

And we also finetune VGG16 in Pascal VOC dataset.

### Running 

A demo has been shown in jupyter notebook.