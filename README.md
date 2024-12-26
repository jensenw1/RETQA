# RETQA: A Large-Scale Open-Domain Tabular Question Answering Dataset for the Real Estate Sector


## :boom: This work has been accepted for publication in AAAI-2025 (acceptance rate 23.4%).

This repository contains the dataset and code released for the paper "RETQA: A Large-Scale Open-Domain Tabular Question Answering Dataset for Real Estate Sector".


RETQA is the first large-scale open-domain Chinese Tabular Question Answering (TQA) dataset focused on the real estate domain. It comprises 4,932 tables and 20,762 question-answer pairs across 16 sub-fields within three major domains: property information, real estate company finance information, and land auction information.

This dataset poses unique challenges for tabular question answering due to its long-table structures, open-domain retrieval requirements, and multi-domain queries. To address these challenges, the paper also introduces the SLUTQA framework, which integrates large language models with spoken language understanding tasks to enhance retrieval and answering accuracy.

![Pipeline](https://github.com/jensenw1/RETQA/blob/main/figures/pipeline.png)

### The repository includes:

* [X] The RETQA dataset in JSON format

* [ ] The source code for the SLUTQA framework

* [ ] Pre-trained SLUTQA model checkpoints

* [ ] Detailed instructions for dataset usage and model training/evaluation

This resource is intended to advance tabular question answering research in the real estate domain and address critical challenges in open-domain and long-table question-answering. We hope it will be a valuable contribution to the research community.



 
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
