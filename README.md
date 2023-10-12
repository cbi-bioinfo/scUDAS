# Cell type prediction for single-cell RNA sequencing utilizing unsupervised domain adaptation and semi-supervised learning
scUDAS is a cell type prediction model for scRNA-seq utilizing unsupervised domain adaptation and semi-supervised learning to reduce the difference in distributions between the datasets. Firstly, to train the classifier, we pre-train the proposed model based on the source dataset which has cell type information. After that, scUDAS is trained on the target dataset, leveraging adversarial training to align the distribution of the target dataset to that of the source dataset. Finally, scUDAS is re-trained to improve the performance through semi-supervised learning by leveraging both the source dataset with ground truth cell types and the target dataset with consistency regularization.

![scUDAS_workflow](https://github.com/cbi-bioinfo/scUDAS/assets/48755108/b24abe98-f9f4-4bf1-a96d-41ddc6358a5c)

## Requirements
* Tensorflow (>= 1.8.0)
* Python (>= 2.7)
* Python packages : numpy, pandas

## Usage
Clone the repository or download source code files.

## Contact
If you have any question or problem, please send an email to **pcr0827@sookmyung.ac.kr**
