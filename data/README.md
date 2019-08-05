# Overview
This project utilizes chest x-ray data from a variety of data sources.

TODO list:
- [ ] Write a shell script (or whatever) to automate the data download process.

## Montgomery County X-ray Set
X-ray images from the Department of Health and Human Services of Montgomery County, MD, United States is available at 
https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/

1. Download data set from [here](http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip).
2. Extract the `zip` file in `data` folder. The folder structure should look something like this:
```bash
data
├── montgomery
│   ├── clinical_readings
│   ├── images
│   └── masks
│       ├── left
│       └── right
├── ...
│   ├── ...
│   └── ...
└── README.md
```


## Shenzhen Hospital X-ray Set
X-ray images from the Shenzhen No.3 Hospital in Shenzhen, Guangdong, China can be found at 
https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/

0. Download data set from [here](http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip).
0. Shenzhen data set does not contain lung segmentation masks. They can be downloaded separately from [here](https://www.kaggle.com/yoctoman/shcxr-lung-mask). To download the data, you will need to sign up for [Kaggle](https://www.kaggle.com)
0. Extract the `zip` files in `data` folder. The folder structure should look something like this:
```bash
data
├── shenzhen
│   ├── clinical_readings
│   ├── images
│   └── masks
├── ...
│   ├── ...
│   └── ...
└── README.md
```


## NIH Chest X-ray Dataset
```
https://www.kaggle.com/nih-chest-xrays/data
```

## CheXpert
```
https://stanfordmlgroup.github.io/competitions/chexpert/
```
