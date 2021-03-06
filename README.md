# PyTorch Implementations of various state of the art architectures. 

## 1. MLP-Mixer: An all-MLP Architecture for Vision (https://arxiv.org/abs/2105.01601)

<img src="./imgs/mlp.PNG" width="500px"></img>

```python
import torch
from MLP_Mixer import MLPMixer

model = MLPMixer(
    classes= 10,
    blocks= 6,
    img_dim= 128,
    patch_dim= 128,
    in_channels= 3,
    dim= 512,
    token_dim= 256,
    channel_dim= 2048
)

x = torch.randn(1, 3, 128, 128)
model(x) # (1, 10)
```

## 2. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (https://arxiv.org/abs/2102.04306)

<img src="./imgs/transunet.PNG" width="500px"></img>

```python
import torch
from TransUNet import TransUNet

model = TransUNet(
    img_dim= 128,
    patch_dim= 16,
    in_channels= 3, 
    classes= 2, 
    blocks= 6,
    heads= 8,
    linear_dim= 1024
)

x = torch.randn(1, 3, 128, 128)
model(x) # (2, 128, 128)
```

## 3. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (https://arxiv.org/abs/2010.11929)

<img src="./imgs/vit.PNG" width="500px"></img>

```python
import torch
from ViT import ViT

model = ViT(
    img_dim= 128,
    in_channels= 3,
    patch_dim= 16,
    classes= 10,
    dim= 512,
    blocks= 6,
    heads= 4,
    linear_dim= 1024,
    classification= True 
)

x = torch.randn(1, 3, 128, 128)
model(x) # (1, 10)
```

## 4. Fastformer: Additive Attention Can Be All You Need (https://arxiv.org/abs/2108.09084)

<img src="./imgs/fastformer.PNG" width="500px"></img>

```python
import torch
from fastformer import FastFormer

model = FastFormer(
    in_dims= 256, 
    token_dim= 512, 
    num_heads= 8
)

x = torch.randn(1, 128, 256)
model(x) # (1, 128, 512)
```

## 5. Single Image Super-Resolution via a Holistic Attention Network (https://arxiv.org/abs/2008.08767)

<img src="./imgs/han.PNG" width="500px"></img>

```python
import torch
from han import HAN

model = HAN(in_channels= 3)

x = torch.randn(1, 3, 256, 256)
model(x) # (1, 3, 512, 512)
```

