
# Reproducing the PyTorch implementation of **CASS** from paper: 

[Efficient Representation Learning for Healthcare with Cross-Architectural Self-Supervision](https://proceedings.mlr.press/v219/singh23a.html)

## Original Source code in:

https://github.com/pranavsinghps1/CASS

# Description:

CASS stands for Cross-Architectural Self-Supervised Learning with its primary aim to work robustly with small batch size and limited computational resources, to make self-supervised learning accessible. The authors of the paper have tested CASS for various label fractions (1%, 10% and 100%), with three modalities in medical imaging (Brain MRI classification, Autoimmune biopsy cell classification and Skin Lesion Classification), various dataset sizes (198 samples, 7k samples and 25k samples) as well as with Multi-class and multi-label classification. 

Some of the data used by the authors are private data and paid data. So we reproduced the paper with BrainMRI and ISIC 2019 Dataset. Since the ISIC 2019 dataset is huge, with our limited computational ability, we were unable to do more experiment with that dataset. So we used BrainMRI dataset as our primary.

## Datasets:

### Brain MRI Classification

Courtesy of [Cheng, Jun] [1] This dataset contains 7k samples of brain MRI with different tumour-related diseases and is a multi-class classification in this context. Train/validation/test splits have already been provided by the dataset curator.

  
### SIIM-ISIC 2019 Dataset

This is a collection of skin lesions images contributed to the [2019 SIIM-ISIC challenge] [2]. This contains 25k samples and is a multi-class classification problem. For train/validation/test splits we follow an 80/10/10 split.

## Dependencies

```

pip install torchmetrics

pip install torchcontrib

pip install pytorch-lightning

pip install timm

pip install tensorboard

```
All the requirements listed by original authors are mentioned in requirements.txt. We ran this code with pytorch 2.0 and installed above mentioned packages.

# Pre-processing

For creating class weights for Focal loss used during downstream fine-tuning, we use the `normalize` function

```
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
  
# assign array and range
array_1d = [1321,1457,1595,1339]
range_to_normalize = (0.2, 1)
normalized_array_1d = normalize(
    array_1d, range_to_normalize[0], 
  range_to_normalize[1])
  
# display original and normalized array
print("Original Array = ", array_1d)
print("Normalized Array = ", normalized_array_1d)

```
All the other preprocessing code were mentioned in DL4H_117.ipynb file.

## Explore files:

-- [DL4H_117.ipynb](https://github.com/ShineyAji/DLH_CASS/blob/main/DL4H_117.ipynb "BrainMRI"):contains the code for self-supervised and downstream supervised fine-tuning for BrainMRI Dataset. It also contains the evaluation code for trained and saved model.

-- [DLH_CASS_ISIC2019.ipynb](https://github.com/ShineyAji/DLH_CASS/blob/main/DLH_CASS_ISIC2019.ipynb "isic"):contains the code for self-supervised and downstream supervised fine-tuning for 2019 ISIC Dataset. 


## References:

-  `[1]:https://figshare.com/articles/dataset/brain_tumor_dataset/1512427` & `https://www.hindawi.com/journals/cin/2022/3236305/`

-  `[2]:https://challenge.isic-archive.com/data/#2019`
