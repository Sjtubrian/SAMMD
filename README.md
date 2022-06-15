# Maximum Mean Discrepancy Test is Aware of Adversarial Attacks
Hi, this is the pytorch code for Semantic-Aware Maximum Mean Discrepancy Test, presented in the ICML2021 paper " Maximum Mean Discrepancy Test is Aware of Adversarial Attacks" (http://arxiv.org/abs/2010.11415). This work is done by  

- Ruize Gao (HKBU,CUHK), ruizegao@cuhk.edu.hk  
- Feng Liu (UTS), feng.liu@uts.edu.au  
- Jingfeng Zhang (RIKEN), jingfeng.zhang@riken.jp  
- Bo Han (HKBU), bhanml@comp.hkbu.edu.hk   
- Tongliang Liu (USYD), tongliang.liu@sydney.edu.au  
- Gang Niu (RIKEN), gang.niu.ml@gmail.com  
- Masashi Sugiyama (RIKEN, U-Tokyo), sugi@k.u-tokyo.ac.jp   
# Software
We recommend users to install python via Anaconda (python 3.7.3), which can be downloaded from https://www.anaconda.com/distribution/#download-section.  

Most codes require freqopttest repo (interpretable nonparametric two-sample test) to implement ME and SCF tests, which can be installed by  
```
pip install git+https://github.com/wittawatj/interpretable-test
```
Torch: 1.1.0  

Python: 3.7.3  

CUDA: 10.1  

# Obtain adversarial data
SAMMD Test can help practitioners to detect adversarial or mixed data generated by any methods.  

We provide some examples of generating adversarial data in the adv folder.  

The CIFAR-10 dataset and the SVHN dataset can be downloaded via Pytorch.  

1) run
```
python train_model.py
```

2) run  
```
python adv_generator.py  
```
you can obtain adversarial data by FGSM, BIM, PGD and CW attack method.

We also provide some exampled trained models and adversarial data. Please download it:
https://drive.google.com/drive/folders/1VtsGXeAKscUBiDT3ty7YxTb6UiHgZ9tu?usp=sharing

# Baselines
- MMD-G: MMD with a Gaussian kernel;  

- MMD-O: MMD with a bandwidth optimized Gaussian kernel;  

- MMD−D: MMD wth a deep kernel;  

- Mean embedding (ME): a test based on differences in Gaussian kernel mean embeddings at a set of optimized points;  

- Smooth characteristic functions (SCF): a test based on differences in Gaussian mean embeddings at a set of optimized frequencies;  

- Classifier two-sample test (C2ST): a special case of MMD, including C2STS-S and C2ST-L;  

run  
```
python Baselines.py  
```
you can obtain average test power of MMD-G, MMD-O, MMD-D, ME, SCF, C2ST-S and C2ST-L Test;  


# Semantic-Aware Maximum Mean Discrepancy (SAMMD) Test
run  
```
python SAMMD_Test.py  
```
you can obtain average test power of SAMMD Test;

# Citation
If you are using this code for your own researching, please consider citing
```
@inproceedings{gao2020maximum,
  title={Maximum Mean Discrepancy is Aware of Adversarial Attacks},
  author={Gao, Ruize and Liu, Feng and Zhang, Jingfeng and Han, Bo and Liu, Tongliang and Niu, Gang and Sugiyama, Masashi},
  booktitle={ICML},
  year={2021}
}
```

# Acknowledgment
RZG and BH were supported by HKBU Tier-1 Start-up Grant and HKBU CSD Start-up Grant. BH was also supported by the RGC Early Career Scheme No. 22200720, NSFC Young Scientists Fund No. 62006202 and HKBU CSD Departmental Incentive Grant. TLL was supported by Australian Research Council Project DE-190101473. JZ, GN and MS were supported by JST AIP Acceleration Research Grant Number JPMJCR20U3, Japan, and MS was also supported by Institute for AI and Beyond, UTokyo. FL would like to thank Dr. Yiliao Song and Dr. Wenkai Xu for productive discussions.


