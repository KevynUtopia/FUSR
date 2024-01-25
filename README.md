# Unpaired Optical Coherence Tomography Angiography Image Super-Resolution via Frequency-Aware Inverse-Consistency GAN

## Overview


This repository contains the offical PyTorch implementation of the paper: "[Unpaired Optical Coherence Tomography Angiography Image Super-Resolution via Frequency-Aware Inverse-Consistency GAN]([https://link.springer.com/chapter/10.1007/978-3-031-16434-7_62](https://arxiv.org/pdf/2309.17269.pdf))"


## Dependences
- Prepare enviornment:
  ```shell script
  conda env create --name fusr --file fusr.yaml
  conda activate fusr
  git clone https://github.com/fbcotter/pytorch_wavelets
  cd pytorch_wavelets
  pip install .
  ```
  
- Training:
  ```shell script
  bash ./run.sh
  ``` 

## Acknowledgements
Wavelets operation is developed from [pytorch_wavelets]([https://github.com/orpatashnik/StyleCLIP](https://github.com/fbcotter/pytorch_wavelets)).


## Citation
Please cite our paper as:
```
@article{zhang2023unpaired,
  title={Unpaired Optical Coherence Tomography Angiography Image Super-Resolution via Frequency-Aware Inverse-Consistency GAN},
  author={Zhang, Weiwen and Yang, Dawei and Che, Haoxuan and Ran, An Ran and Cheung, Carol Y and Chen, Hao},
  journal={arXiv preprint arXiv:2309.17269},
  year={2023}
}
```
