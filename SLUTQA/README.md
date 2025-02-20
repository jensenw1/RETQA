This directory contains the implementation of the entire SLUTQA framework. The directory structure is as follows:
```
.
├── BERT-fine-tune.ipynb
├── prompts
│   ├── first_prompts.json
│   ├── retrival_prompt.json
│   ├── second_prompts.json
│   ├── SLU_prompt.json
│   └── SQL_prompt.json
├── results
├── SLUTQA(BERT).ipynb
├── SLUTQA(ICL).ipynb
├── table_names.json
└── utils.py

3 directories, 10 files
```

The `prompts` directory contains all the prompt templates; for detailed explanations, refer to the paper.  
The `results` directory stores intermediate results.  
`table_names.json` is a collection of all table names in the database, used for BM25 matching.