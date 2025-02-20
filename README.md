# RETQA: A Large-Scale Open-Domain Tabular Question Answering Dataset for the Real Estate Sector


## :boom: This work has been accepted for publication in AAAI-2025 (acceptance rate 23.4%).

This repository contains the dataset and code released for the paper "RETQA: A Large-Scale Open-Domain Tabular Question Answering Dataset for Real Estate Sector".


RETQA is the first large-scale open-domain Chinese Tabular Question Answering (TQA) dataset focused on the real estate domain. It comprises 4,932 tables and 20,762 question-answer pairs across 16 sub-fields within three major domains: property information, real estate company finance information, and land auction information.

This dataset poses unique challenges for tabular question answering due to its long-table structures, open-domain retrieval requirements, and multi-domain queries. To address these challenges, the paper also introduces the SLUTQA framework, which integrates large language models with spoken language understanding tasks to enhance retrieval and answering accuracy.

![Pipeline](https://github.com/jensenw1/RETQA/blob/main/figures/pipeline.png)

## Requirements and Installation

If you want to get the same result with the paper, please follow below
```
torch==2.3.1
numpy==1.26.4
transformers==4.41.2
pandas==2.2.2
ipywidgets==8.1.3
sklearn==1.5.1
langchain==0.2.6
langchain_openai==0.1.14
langchain_community==0.2.6
langchain_core==0.2.11
rank_bm25==0.2.2
jieba==0.42.1
psycopg2==2.9.9
```

## Usage


### Directory structure

```
.
├── database
│   ├── import_table.ipynb
│   ├── markdown_tables
│   ├── READEME.md
│   └── tables.tar
├── datasets
│   ├── README.md
│   ├── test.json
│   ├── train.json
│   └── validation.json
├── figures
├── README.md
└── SLUTQA
    ├── BERT-fine-tune.ipynb
    ├── model
    ├── prompts
    ├── README.md
    ├── results
    ├── SLUTQA(BERT).ipynb
    ├── SLUTQA(ICL).ipynb
    ├── table_names.json
    └── utils.py

9 directories, 14 files
```

### Importing Data  

1. **Create a New Database**  
   Start by creating a new database. Here, we use a PostgreSQL database deployed via a Docker container. (Please ensure Docker is already installed on your system before proceeding.)  
```docker
docker run -id \
--name=re-postgres \
-v ./data:/var/lib/postgresql/data \
-p 25432:5432 \
-e POSTGRES_PASSWORD='123456' \
-e POSTGRES_USER='postgre' \
-e LANG=C.UTF-8 \
--restart=always \
postgres:alpine
```
After successful execution, the database will be accessible via port `25432`.
2. **Read and Import Tables**  
   Read all the tables located in the `tables` directory, and import them into the newly created database.  
   You can run the script `databases/import_table.ipynb` to complete the database setup. Make sure to update it with the PostgreSQL username and password you have configured.



### The repository includes:

* [X] The RETQA dataset in JSON format

* [X] The source code for the SLUTQA framework

* [ ] Pre-trained SLUTQA model checkpoints

* [X] Detailed instructions for dataset usage and model training/evaluation

This resource is intended to advance tabular question answering research in the real estate domain and address critical challenges in open-domain and long-table question-answering. We hope it will be a valuable contribution to the research community.

###  This repository is still under construction:  

* [X] All tables will be uploaded by February 10, 2025.
    
* [X] All SLUTQA codes will be uploaded by February 15, 2025.

* [ ] If you have any questions, please contact **jensenwang@mail.bnu.edu.cn**.

 
The acquisition of real estate data is supported by [Elmleaf Ltd.\(Shanghai\)](https://www.elmleaf.com.cn/dataset) : https://www.elmleaf.com.cn/dataset.




# Citation
If you found this work useful for you, please consider citing it.
```
@inproceedings{
aaai2025retqa,
title={{RETQA}: A Large-Scale Open-Domain Tabular Question Answering Dataset for Real Estate Sector},
author={Zhensheng Wang and Wenmian Yang and Kun Zhou and Yiquan Zhang and Weijia Jia},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2024},
url={https://arxiv.org/abs/2412.10104}
}
```
