# DivSR

This is our Tensorflow implementation for Leave No One Behind: Enhancing Diversity While Maintaining Accuracy in Social Recommendation

# Requirements
Environments:
```
python==3.8
tensorflow==2.4.0
```
Install environments by:
```
pip install -r requirements.txt
```


# Quick Start

## train a SocialRS (w/o social) as the teacher model (e.g. DiffNet)
```
python entry.py --data_name yelp --model_name kd_diffnet --social 0
```

## train a SocialRS as the student model (e.g. KD-DiffNet)
```
python entry.py --data_name yelp --model_name kd_diffnet --kd 1 --social 1 --gamma 1.0
```

# Output
Output path will be like this:
```
Results/
    - yelp/
        - kd_diffnet/
            - kd/
                user_embed.npy
                item_embed.npy
            - base/
            --without/
```

# Dataset
## Get our prepared dataset
We have placed the processed data in the /data directory.

Social backbones (TrustMF, SocialMF, DiffNet, MHCN), are developed based on [QRec](https://github.com/Coder-Yu/QRec).
DESIGN is developed based on [DESIGN](https://www.dropbox.com/s/uqmsr67wqurpnre/Supplementary%20Material.zip?dl=0)
