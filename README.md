# DivSR

This is our Tensorflow implementation for Enhancing Diversity on Social Recommendation with Knowledge Distillation

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
python main.py --data_name yelp --model_name kd_diffnet --social 0
```

## train a SocialRS as the student model (e.g. DiffNet)

Train Ada-Ranker in the second-stage.
You need fit the `is_adaretrieval=1` and `load_best_model=1` to download your pre-trained model in the first stage
(notice that the model_file path should be consistent with the generated model in the first stage)

```
python main.py --data_name yelp --model_name kd_diffnet --kd 1 --social 1 --gamma 1.0
```

See more details of main files in `Main/`.

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
