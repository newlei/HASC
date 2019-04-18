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

  > <user, rating,item> matrix, 
  >
  > Upload History  <user, upload history item> matrix
  >
  >  Social Neighborhood  <user, follow user> matrix    
  >
  > Owner Admiration  <item, owner user> matrix
  >
  > Item Visual Information <item, image visual feature> matrix
  