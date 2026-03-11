# CMMC: A Novel Relation Extraction Architecture
## Description
The code implementation of the CMMC. You need to add the data folder yourself to avoid the code reporting errors. The environment configuration for running the reference is:
NVIDIA GeForce RTX 2080 Ti --- CUDA 12.9 --- Torch 1.12.0 --- Python 3.8.20
## Dataset Information
+ The SemEval-2010 Task 8 dataset is publicly accessible at available at https://docs.google.com/document/u/0/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/mobilebasic?tab=t.0&_immersive_translate_auto_translate=1.
+ The RETA is a non-public dataset requiring access permission: https://catalog.ldc.upenn.edu/LDC2018T24.
+ After obtaining the data, it needs to be consolidated into the following format: {"sentence": "tom thabane resigned in october last year to form the all basotho convention -lrb- abc -rrb- , crossing the floor with 17 members of parliament , causing constitutional monarch king letsie iii to dissolve parliament and call the snap election .", "label": 3, "e1": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80], "e1_begin": 10, "e2": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90], "e2_begin": 0}
+ For data processing code, please refer to: preprocess/preprocess.py
## Code Information
Description of the project's primary documents
+ train.py: Entry file for the project.
+ requirements.txt: Dependency environment for project operation.
+ data_w2v.py: Code for loading Word2vec word embedding.
+ data_fasttext.py: Code for loading FastText word embedding.
+ data.py:  Code for loading BERT word embedding.
+ data_distilbert.py: Code for loading DistilBERT word embedding.
+ evaluate.py: Code related to performance evaluation.
+ wordvec: Code related to static word vectors.
+ utils: Toolkit.
+ preprocess: Code for data preprocessing.
+ prepare_data: Related content on relation mapping.
+ model: Code related to the network.
+ experiments: Includes relevant content pertaining to experimental parameters.
+ get_ext_res: Code for integrating experimental results.
## Usage Instructions
+ Step 1: Create a virtual Python environment using Anaconda.
+ Step 2: Run `pip install -r requirements.txt` to install environment dependencies.
+ Step 3: Once ready, run the code using `python train.py`.
## Requirements
Please refer to the requirements.txt. The specific installation method is as follows: `pip install -r requirements.txt`.
## Meghodology
Please refer to the manuscript.
## Additional Information
If you have any other questions, please feel free to contact us!
