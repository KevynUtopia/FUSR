# Frequency-Aware Inverse-Consistent Deep Learning for OCT-Angiogram Super-Resolution

This is the official PyTorch implementation of "[Frequency-Aware Inverse-Consistent Deep Learning for OCT-Angiogram Super-Resolution](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_62)", MICCAI 2022.  
Please follow `main.ipynb` to execute the code.  

## Illustration of our method:  

> Our generators  
<img src="./demo/img01.jpg" width = "500" alt="Generators" align="center" />  
  
> Our Discriminators  
<img src="./demo/img02.jpg" width = "500" alt="Discriminators" align="center" />  

## Access to the pretrained model:  
Create a folder `pre_trained` and put pretrained model in it. Download pretrained model [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wzhangbu_connect_ust_hk/Ev6yRUKDKytKmWjZwxWaML4BqodCQjg6U9EntuPnjztyLw?e=1crfOa).  

## Excution
```
python train.py --l2h_hb 13 --l2h_lb 27 --h2l_hb 13 --h2l_lb 27 --beta1 15 --beta2 15 --beta3 5 --beta4 5 --n_epochs 50 --dis_iter 2 --batchSize 1 --hfb 0.8 --n_layers 5
python train.py --n_epochs 50 --l2h_hb 17 --l2h_lb 13 --h2l_hb 17 --h2l_lb 13 --dis_iter 1 --hfb 1.0 --input_D_nc 1 --ffl_h --ffl_l --loss_weight 10.0
```
