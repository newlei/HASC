# HASC

A Hierarchical Attention Model for Social Contextual Image Recommendation
We develop a hierarchical attention model for social contextual image recommendation. In addition to basic latent user interest modeling in the popular matrix factorization based recommendation, we identify three key aspects (i.e., upload history, social influence, and owner admiration) that affect each user’s latent preferences, where each aspect summarizes a contextual factor from the complex relationships between users and images. After that, we design a hierarchical attention network that naturally mirrors the hierarchical relationship (elements in each aspects level, and the aspect level) of users’ latent interests with the identified key aspects. 

We provide TensorFlow implementations for HASC model.

**Note**: The current software works well with Tensorflow 0.14+. 

## Prerequisites

- TensorFlow
- Python 2.7
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/newlei/HASC.git
cd HASC
```

### Dataset

- Need :

  > Favor <user,item> matrix,  e.g.:<1,2> is that user(id:1) favors item(id:2) 
  >
  > Upload History  <user, upload history item> matrix, e.g.:<1,[7,11,23]> show that user(id:1) uploads some items(id:7,11 and 23)  
  >
  >  Social Neighborhood  <user, follow user> matrix, e.g.:<1,[5,6]> show that user(id:1) follows some users(id:5,6)   
  >
  > Owner Admiration  <item, owner user> matrix, e.g:<7,1> show that items(id:7) is uploaded by user(id:1) 
  >
  > Item Visual Information <item, image visual feature> matrix, e.g:<2,[\*]\*4096> is the image feature of item(id:2) 

- For the data in this paper, please seed mail to me：
  >  Please indicate the required dataset and transmission method in the email.

### Train/test

- Train a model:

```python
#!./HASC
python HASC_model.py
```


