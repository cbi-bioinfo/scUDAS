# Cell type prediction for single-cell RNA sequencing utilizing unsupervised domain adaptation and semi-supervised learning
scUDAS is a cell type prediction model for scRNA-seq utilizing unsupervised domain adaptation and semi-supervised learning to reduce the difference in distributions between the datasets. Firstly, to train the classifier, we pre-train the proposed model based on the source dataset which has cell type information. After that, scUDAS is trained on the target dataset, leveraging adversarial training to align the distribution of the target dataset to that of the source dataset. Finally, scUDAS is re-trained to improve the performance through semi-supervised learning by leveraging both the source dataset with ground truth cell types and the target dataset with consistency regularization.

![scUDAS_workflow](https://github.com/cbi-bioinfo/scUDAS/assets/48755108/b24abe98-f9f4-4bf1-a96d-41ddc6358a5c)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas

## Usage
Clone the repository or download source code files.

1. Edit **"run_scUDAS.sh"** file having dataset files for source and target. Modify each variable values in the bash file with filename for your own dataset. Each file should contain the header and follow the format described as follows :

- ```source_X, target_X``` : File with a matrix or a data frame containing gene expression, where each row and column represent **sample** and **feature**, respectively. Example for dataset format is provided below.

```
PTPN2,GTF2A2,...,HN1L,ATHL1
2.1100,0.0,...,0.0,3.7135
...
```

- ```source_Y, target_Y``` : File with a matrix or a data frame contatining cell-type for each sample, where each row represent **sample**. Cell-type names used for source and target should be included and users should label each cell-type as 1 and 0 for others in the same order in source dataset to be matched. Example for data format is described below.

```
B cell,CD14+ monocyte,CD4+ T cell,Cytotoxic T cell,Dendritic cell,Megakaryocyte,Plasmacytoid dendritic cell
0,0,0,1,0,0,0
0,0,0,1,0,0,0
0,1,0,0,0,0,0
0,0,1,0,0,0,0
...
```

2. Use **"run_scUDAS.sh"** to predict cell tyeps subtypes for single-cell RNA sequencing.

3. You will get an output **"prediction.csv"** with predicted cell types for target dataset.

## Contact
If you have any question or problem, please send an email to **pcr0827@sookmyung.ac.kr**
