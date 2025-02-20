import numpy as np
import pandas as pd
from ipywidgets import FloatProgress
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import json
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from tqdm import tqdm
import ast
import re
import math
from collections import Counter, defaultdict
import re
from rank_bm25 import BM25Okapi
import jieba
import psycopg2


# 将生成的表格名称存储到json文件当中
class PredictResultStorage:
    def __init__(self, file_name='testdata.json'):
        self.current_data = {}
        self.file_name = file_name

    def set_predict_slots_name(self, predict_slots_name):
        self.current_data["predict_slots_name"] = predict_slots_name
        
    def set_true_slots_name(self, true_slots_name):
        self.current_data["true_slots_name"] = true_slots_name

    def set_predict_intent_name(self, predict_intent_name):
        self.current_data["predict_intent_name"] = predict_intent_name
        
    def set_true_intent_name(self, true_intent_name):
        self.current_data["true_intent_name"] = true_intent_name

    def set_uuid(self, uuid):
        self.current_data["uuid"] = uuid

    def save_data(self):
        """将数据保存到 JSON 文件"""
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r+') as f:
                data = json.load(f)
                data.append(self.current_data)
                f.seek(0)
                json.dump(data, f, indent=4)
        else:
            with open(self.file_name, 'w') as f:
                json.dump([self.current_data], f, indent=4)
        # 清空 current_data，以便存储下一条数据
        self.current_data = {}


# 将slots转化为关键词
def restore_keywords_from_query(query, slots):
    keywords = []
    current_tokens = []
    current_label = None
    query = list(query)
    if isinstance(slots, str):
        slots = slots.split(' ')
    if slots[0] == '[CLS]':
        slots = slots[1:-1]

    for token, slot in zip(query, slots):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))
    keyword_pair = []
    for keyword in keywords:
        if keyword[-1] == 'city':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'district':
            keyword_pair.append(f'区域:{keyword[0]}')
        elif keyword[-1] == 'community':
            keyword_pair.append(f'项目名称:{keyword[0]}')
        elif keyword[-1] == 'enterprise':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'年份:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'月份:{keyword[0]}')
        elif keyword[-1] == 'land':
            keyword_pair.append(f'地块名称:{keyword[0]}')

    return keyword_pair


def extract_intent_and_slots(text):
    # Regular expression patterns to extract intent and slots
    intent_pattern = r"intent[:>)]?\s*(.*?)(?=[<\(]_?slots_?[>\)])"
    slots_pattern = r"[<\(]_?slots_?[>\)]\s*(\[.*?\])"
    #intent_pattern = r"intent[:>)]?\s*([^,]+?)(?=[<\(]_?slots_?[:>)]?)"
    #slots_pattern = r"[<\(]_?slots_?[:>)]?\s*(\[.*?\])"

    # Extracting intent
    intent_match = re.search(intent_pattern, text)
    if intent_match:
        intent = intent_match.group(1).strip()
    else:
        intent = None

    # Extracting slots
    slots_match = re.search(slots_pattern, text)
    if slots_match:
        slots_str = slots_match.group(1)
        try:
            slots_list = ast.literal_eval(slots_str)  # Safely evaluates the string as a list
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing slots: {e}")
            slots_list = None
    else:
        slots_list = None

    return intent, slots_list

def slots_to_dict(slots_list):
    slots_dict = {}
    if slots_list:
        for slot in slots_list:
            key, value = slot.split(':')
            slots_dict[key.strip()] = value.strip()
    return slots_dict



def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengthes not equal!'
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    for true_label, pred_label in zip(trues, preds):
        # 将列表元素全部处理为list
        if isinstance(true_label, type('')):
            true_label = [true_label]
        if isinstance(pred_label, type('')):
            pred_label = [pred_label]
        # 查准率
        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        # 查全率
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    P = precision/precision_all
    R = recall/recall_all
    F1 = 2 * P * R / (P + R)
    return P, R, F1


def fix_missed_bracket_indices(input_str):
    brackets_indices = []
    for index, char in enumerate(input_str):
        if char in ('(', ')'):
            brackets_indices.append(index)
    #print(brackets_indices)
    pair = []
    if len(brackets_indices)%2 != 0:
        items = [input_str[i] for i in brackets_indices]
        #print(items)
        for i, index in enumerate(brackets_indices):
            #print(i)
            if input_str[index]=='(' and input_str[brackets_indices[i+1]]==')':
                pair.append(True)
            elif input_str[index]==')' and input_str[brackets_indices[i-1]]=='(':
                pair.append(True)
            else:
                pair.append(False)
    for index, j in enumerate(pair):
        if not j:
            if items[index] == '(':
                insert_index = brackets_indices[index+1]
                input_str = input_str [:insert_index-1] + ')' + input_str [insert_index-1:] 
            
    return input_str

def extract_content(input_string):
    last_parenthesis_index = input_string.rfind(')')
    if last_parenthesis_index != -1:
        return input_string[:last_parenthesis_index]
    else:
        return input_string  # 如果没有找到')'，返回原始字符串


# execute_sql()输入SQl即可返回数据库执行结果
class PostgresQueryExecutor:
    def __init__(self, host='10.10.1.237', database='价格查询', user='postgres', password='123456', port="25432"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cur = self.conn.cursor()
        except DatabaseError as e:
            print(f"Database connection error: {e}")
            raise

    def execute_sql(self, sql_statement):
        if not self.conn or not self.cur:
            self.connect()
        try:
            self.cur.execute(sql_statement)
            result = self.cur.fetchall()  # 获取执行结果
            headers = [desc[0] for desc in self.cur.description]
            self.conn.commit()  # 提交事务
            
            return headers, result
        except Exception as e:
            print(f"Error executing SQL statement: {e}")
            return None, None

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
house_sales_field  = ['项目成交状况查询', '项目开发商信息查询', '小区成交均价查询']
land_sales_field  = ['小区绿化率查询', '建筑密度查询', '容积率查询', '地块总价查询', '地块归属查询', '地块成交时间查询']
enterprise_sales_field  = ['企业营业成本查询', '企业性质查询', '企业风险查询', '企业负债查询', '营业总收入查询', '营业利润查询', '企业债务违约查询']


import traceback
def table2markdown(QA: dict):
    if '+' in QA['intent']:
        intent = QA['intent'].split('+')
    else:
        intent = [QA['intent']]
    if set(intent).issubset(house_sales_field):
        dbname = '价格查询'
        SQL_template = 'SELECT "项目名称","区域","成交均价","时间（月份）","成交套数","集团开发商","年月" FROM '
    elif set(intent).issubset(land_sales_field):
        dbname = '土地资产'
        SQL_template = 'SELECT "地块名称","项目名称","成交日期","所属集团","成交总价","容积率","建筑密度(%)","绿化率(%)","城市区域","时间(月份)" FROM '
    elif set(intent).issubset(enterprise_sales_field):
        dbname = '企业财务'
        SQL_template = 'SELECT "企业名称","年份","营业总收入","营业利润","营业总成本","资产总计","负债合计","是否国有","信用债情况","风险等级" FROM '

    #table_answer_SQL = QA['SQL_query'][:7] + '"项目名称",' + QA['SQL_query'][7:]
    table_names = QA['chosed_table_name']
    sql_statements = []
    if isinstance(table_names, str):
        table_names = [table_names]
    for table_name in table_names:
        sql_statements.append(SQL_template + '"' + table_name + '"' + ';')
    #print(sql_statements)
    markdown_answers = []
    for sql_statement, table_name in zip(sql_statements, table_names):
        try:
            executor = PostgresQueryExecutor(database=dbname)
            table_heads, table_results = executor.execute_sql(sql_statement)
            #print(f'table_heads:{table_heads}')
            #print(f'table_results:{table_results}')
            markdown_heads = '<table_name> ' + str(table_name) + ' ' + 'col :'
            for i in table_heads:
                markdown_heads = markdown_heads + ' ' + i + ' '+ '|'
            markdown_heads = markdown_heads[:-1]
            markdown_answer = markdown_heads
            for i, row in enumerate(table_results):
                rows = f'row {i+1} :'
                for item in row:
                    item = str(item)
                    rows = rows + ' ' + item + ' ' + '|'
                rows = rows[:-1]
                markdown_answer = markdown_answer + rows
            markdown_answer = markdown_answer[:-1]
            markdown_answers.append(markdown_answer)
        
        except Exception as e:
            traceback.print_exc()
            print('执行SQL出错！！！！！！！！！！！')
            print(QA)
    markdown_table = ''
    for markdown_answer in markdown_answers:
        markdown_table = markdown_table + markdown_answer + ' '
    markdown_table = markdown_table[:-1]
    return markdown_table



def align_tokens_with_query(tokens, query):
    query = list(query)
    new_tokens = []
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        elif token == query[0]:
            new_tokens.append(token)
            query = query[1:]
        elif '##' in token:
            token = token[2:]
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    
        elif '[UNK]' == token:
            end_index = query.index(tokens[i+1])
            unk = ''.join(query[0:end_index])
            new_tokens.append(unk)
            query = query[end_index:]

        elif len(token)>1:
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
    return new_tokens    

def restore_keywords_from_tokens(tokens, token_slot):
    keywords = []
    current_tokens = []
    current_label = None
    token_slot = token_slot[1:-1]

    for token, slot in zip(tokens, token_slot):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))
    keyword_pair = []
    for keyword in keywords:
        if keyword[-1] == 'city':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'district':
            keyword_pair.append(f'区域:{keyword[0]}')
        elif keyword[-1] == 'community':
            keyword_pair.append(f'项目名称:{keyword[0]}')
        elif keyword[-1] == 'enterprise':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'年份:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'月份:{keyword[0]}')
        elif keyword[-1] == 'land':
            keyword_pair.append(f'地块名称:{keyword[0]}')

    return keyword_pair

# 将slots转化为关键词
def restore_keywords_from_query(query, slots):
    keywords = []
    current_tokens = []
    current_label = None
    query = list(query)
    if isinstance(slots, str):
        slots = slots.split(' ')
    if slots[0] == '[CLS]':
        slots = slots[1:-1]

    for token, slot in zip(query, slots):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))
    keyword_pair = []
    for keyword in keywords:
        if keyword[-1] == 'city':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'district':
            keyword_pair.append(f'区域:{keyword[0]}')
        elif keyword[-1] == 'community':
            keyword_pair.append(f'项目名称:{keyword[0]}')
        elif keyword[-1] == 'enterprise':
            keyword_pair.append(f'城市:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'年份:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'月份:{keyword[0]}')
        elif keyword[-1] == 'land':
            keyword_pair.append(f'地块名称:{keyword[0]}')

    return keyword_pair



def intent2label(intents_row):
    intents_num = {'企业性质查询': 0,
          '营业利润查询': 1,
          '企业负债查询': 2,
          '项目成交状况查询': 3,
          '企业营业成本查询': 4,
          '小区绿化率查询': 5,
          '建筑密度查询': 6,
          '小区成交均价查询': 7,
          '营业总收入查询': 8,
          '地块总价查询': 9,
          '地块成交时间查询': 10,
          '企业债务违约查询': 11,
          '容积率查询': 12,
          '企业风险查询': 13,
          '地块归属查询': 14,
          '项目开发商信息查询': 15
          }
    intents_label = [0] * len(intents_num)
    if '+' in intents_row:
        intents = intents_row.split('+')
        for intent in intents:
            intents_label[intents_num[intent]] = 1
    elif '+' not in intents_row:
        intent = intents_row
        intents_label[intents_num[intent]] = 1
    return intents_label

def extract_keywords(tokens, token_slot):
    keywords = []
    current_keyword = ''
    current_slot = ''
    
    for token, slot in zip(tokens[1:-1], token_slot[1:-1]):  # Skip [CLS] and [SEP]
        if slot.startswith('B-'):
            if current_keyword:
                keywords.append((current_keyword.strip(), current_slot.split('-')[1]))
            current_keyword = token
            current_slot = slot
        elif slot.startswith('I-') and slot[2:] == current_slot[2:]:
            current_keyword += token
        elif slot == 'O':
            if current_keyword:
                keywords.append((current_keyword.strip(), current_slot.split('-')[1]))
                current_keyword = ''
                current_slot = ''
    
    if current_keyword:
        keywords.append((current_keyword.strip(), current_slot.split('-')[1]))
    
    return keywords


def top2_indices(tensor):
    # 确保tensor至少有两个元素
    if len(tensor) < 2:
        raise ValueError("Input tensor must have at least 2 elements")
    # 使用torch.topk获取top2的值和索引
    _, indices = torch.topk(tensor, k=2, dim=0)
    return indices
    
def find_key(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

