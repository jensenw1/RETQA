Detailed explanation of the files in this directory.
```
.
├── import_table.ipynb
├── markdown_tables
│   └── all_markdown_table.json
├── READEME.md
└── tables.tar

2 directories, 4 files
```

Please execute the following command to generate all the tables in CSV format:
```
tar -xzvf tables.tar.gz
```


## 1. If you want to use SQL for QA  
The `tables` directory contains all tables from three different domains. To simulate a real-world table QA environment, you can store all these tables in a database.  
The following example demonstrates how to store all tables into a PostgreSQL database.
###### a. Create a new PostgreSQL database.
```
docker run -id \
--name=time-series-postgresql \
-v ./data:/var/lib/postgresql/data \
-p 25432:5432 \
-e POSTGRES_PASSWORD='your_password' \
-e POSTGRES_USER='your_name' \
-e LANG=C.UTF-8 \
--restart=always \
postgres:alpine
```
###### b. Run all the code in the `import_table.ipynb` file. This will read all the tables from the `tables` directory, connect to PostgreSQL, create three new databases, and write these tables into them.

## 2. If you want to directly provide markdown-formatted tables to the LLM  
The `markdown_tables/` directory contains all tables in markdown format. If you want to use markdown-formatted tables as input, please read the `all_markdown_table.json` file.