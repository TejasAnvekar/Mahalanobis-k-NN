# Mahalanobis k-NN: A Statistical Lens for Robust Point-Cloud Registrations @ [ImageQuality2025](https://wacv2025-image-quality-workshop2.github.io/index.html)

### [Tejas Anvekar](https://tejasanvekar.github.io/), [Shivanand Venkanna Sheshappanavar](https://sheshap.github.io/)

<div align="left">
<a><img src="./assets/Images/asu_logo.png"  height="70px" ></a>
<a><img src="./assets/Images/uwyo_logo.png"  height="70px" ></a>
</div>

[[arXiv]](https://arxiv.org/abs/2409.06267)
<!-- [[Paper]](https://openaccess.thecvf.com/content/CVPR2023W/DLGC/papers/Anvekar_GPr-Net_Geometric_Prototypical_Network_for_Point_Cloud_Few-Shot_Learning_CVPRW_2023_paper.pdf) -->
<br><br>


<div align="center"> 


<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>  -->


![Teaser](https://github.com/TejasAnvekar/Mahalanobis-k-NN/blob/main/assets/Images/Teaser.png)
The visual supremacy of our proposed methodologies, MDCP-v1 and MDeepUME, becomes apparent in a point cloud registration, involving a target point cloud with only half the points compared to the source point cloud. In the visualization, regions highlighted in red illustrate the limited performance of DCP-v1  and DeepUME , while the regions highlighted in green demonstrate the resilience of the proposed Mahalanobis versions of DCP-v1 and DeepUME.




![Main](https://github.com/TejasAnvekar/Mahalanobis-k-NN/blob/master/assets/Images/main.png)
Illustration highlights the distinction between the Red gradients representing Euclidean distance fields that capture spatial neighbors and the Blue gradients depicting Mahalanobis distance fields, which consider neighbors concerning the underlying data distribution. To further emphasize the point, we present the influence of both Euclidean and Mahalanobis k-NN on a chair point cloud. The black query point is surrounded by Euclidean neighbors shown in red points and Mahalanobis neighbors in blue points. The depiction clearly illustrates the impact of Mahalanobis distance, effectively capturing surficial points per the data distribution vital for precise feature matching.




</div>
<br><br>

<!-- [[Project Webpage](https://nekrasov.dev/mix3d)] [[arXiv](https://arxiv.org/abs/2110.02210)] [[Video](https://mix3d-demo.nekrasov.dev/)]  -->




## Installation

```bash
# step 1. clone this repo
git clone https://github.com/TejasAnvekar/Mahalanobis-k-NN.git
cd Mahalanobis-k-NN

# step 2: install libs step by step
conda create -n MahKNN python=3.8 -y
conda activate MahKNN
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3.1 -c pytorch -y
pip install -r requirements.txt
```





## Train M-DCP (Mahalanobis-KNN)
```bash
cd dcp
#DCP-v1
python main_mah.py --exp_name=mdcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

#DCP-v2 
python main_mah.py --exp_name=mdcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd
```


## Test M-DCP (Mahalanobis-KNN)
```bash
cd dcp
#DCP-v1
python main_mah.py --exp_name=mdcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy --unseen=True

#DCP-v2 
python main_mah.py --exp_name=mdcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy --unseen=True
```





## Train M-DeepUME (Mahalanobis-KNN)
```bash
cd DeepUME
python main_mah.py --exp_name=m-deepume --noise=sampling
```

## Test M-DeepUME (Mahalanobis-KNN)
```bash
cd DeepUME
python main_mah.py --exp_name=m-deepume --eval --noise=zero_intersec --test_dataset=FAUST --model_path=xx/yy
```



## Train DCP
```bash
cd dcp
#DCP-v1
python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

#DCP-v2 
python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd
```


## Test DCP
```bash
cd dcp
#DCP-v1
python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy --unseen=True

#DCP-v2 
python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy --unseen=True
```



## Train DeepUME
```bash
cd DeepUME
python main.py --exp_name=deepume --noise=sampling
```

## Test DeepUME
```bash
cd DeepUME
python main.py --exp_name=deepume --eval 
##or
python main.py --exp_name=pretrained --eval --pretrained='pretrained/deepume.t7' --noise=zero_intersec --test_dataset=FAUST
```



## Re-Create Results of DCP
```bash
cd dcp
# recreate main results (vanilla vs Mahalanobis-KNN)
sh reproduce_main.sh

# recreate few-shot results (vanilla vs Mahalanobis-KNN)
sh reproduce_fsl.sh

# recreate linear svm results (vanilla vs Mahalanobis-KNN)
sh reproduce_linear.sh
```



## Re-Create Results of DeepUMP
```bash
cd DeepUME
# recreate ablation results (vanilla vs Mahalanobis-KNN)
python run_inference.py 
python run_inference.py --mah

```





## Acknowledgment

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Deep Closest Point: Learning Representations for Point Cloud Registration](https://github.com/WangYueFt/dcp), and
[DeepUME: Learning the Universal Manifold Embedding for Robust Point Cloud Registration](https://github.com/langnatalie/DeepUME).



## BibTeX
Please cite our paper if it is helpful to your research:

```
@misc{anvekar2025mahalanobisknnstatisticallens,
      title={{Mahalanobis k-NN: A Statistical Lens for Robust Point-Cloud Registrations}}, 
      author={{Tejas Anvekar and Shivanand Venkanna Sheshappanavar}},
      year={2025},
      eprint={2409.06267},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.06267}, 
}
```
---
