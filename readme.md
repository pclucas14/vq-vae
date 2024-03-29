## Vector Quantized VAEs 

A Pytorch implementation of [this paper](https://arxiv.org/abs/1711.00937)

This repository builds off [this one](https://github.com/rosinality/vq-vae-2-pytorch/)

Running the following command gets ~ 5.05 BPD ( Instead of 4.7 / 4.8, as is reported in said papers [1](http://bayesiandeeplearning.org/2017/papers/54.pdf) [2](https://arxiv.org/pdf/1805.11063.pdf)

```
python main.py --hH 16 --n_codebooks 2 --embed_dim 256 --n_res_channels 256 --n_channels 256 --batch_size 256 --lr 5e-4
```

### Contribute
All contributions / comments / remarks are highly welcomed. 


