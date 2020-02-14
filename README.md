# Graph Convolutional Network
I built a Graph Convolutional Network (GCN) based on the paper by _[Kipf and Welling(ICLR 2017)](http://arxiv.org/abs/1609.02907)_. The original paper is implemented with Tensorflow, whereas I build the GCN on top of PyTorch instead.

The mathematics underlying spectral graph convolutions are explained excellently in [this Medium article](https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801).

In the notebook I implement the  same 2-layer GCN architecture outlined by _Kipf and Welling_ . However, here the GCN is used for fully supervised classification of graphs belonging to different structural classes.

## Requirements
* `numpy`
* `matplotlib`
* `dgl`
* `tqdm`
* `networkx`
* `scipy`
* `torch`
* `sklearn`
