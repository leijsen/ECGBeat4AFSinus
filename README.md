# ECGBeat4AFSinus
A Deep Learning Method for Beat-Level Risk Analysis and Interpretation of Atrial Fibrillation Patients during Sinus Rhythm. 📃[Read the paper](https://doi.org/10.1016/j.bspc.2024.107028)

>**A Deep Learning Method for Beat-Level Risk Analysis and Interpretation of Atrial Fibrillation Patients during Sinus Rhythm** \
>*Biomedical Signal Processing and Control* \
>Jun Lei, Yuxi Zhou, Xue Tian, Qinghao Zhao, Qi Zhang, Shijia Geng, Qingbo Wu, Shenda Hong

*Last update on 26 December 2024*

# Dataset
You could get dataset at [https://www.physionet.org/content/cpsc2021/1.0.0/](https://www.physionet.org/content/cpsc2021/1.0.0/)

## Run Project
1. To modify the dataset path in *'My_util.py'*.
2. `python train_net1d.py`

## Main dependencies
```
python==3.8.17
pytorch==1.13.0
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.10.1
pandas==1.5.3
tqdm==4.65.0
```

## Create an environment 
Use the following command to create an environment based on the *'flowers_env.yml'* file

`conda env create -f flowers_env.yml`

`conda activate flowers_env`


## Reference
We appreciate your citations if you find our paper related and useful to your research!

```
@article{lei2025deep,
  title={A Deep Learning Method for Beat-Level Risk Analysis and Interpretation of Atrial Fibrillation Patients during Sinus Rhythm},
  author={Lei, Jun and Zhou, Yuxi and Tian, Xue and Zhao, Qinghao and Zhang, Qi and Geng, Shijia and Wu, Qingbo and Hong, Shenda},
  journal={Biomedical Signal Processing and Control},
  volume={100},
  pages={107028},
  year={2025},
  publisher={Elsevier}
}
```
