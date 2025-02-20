This directory contains all the QA pairs.

Each QA pair consists of 8 key-value pairs, which are described as follows:  
- **query**: Refers to the natural language question provided as user input.  
- **intent**: The intent corresponding to the natural language question.  
- **slots**: The BIO tagging for the natural language question, represented as strings for annotation purposes.  
- **table_caption_label**: A list or string storing the gold-standard table titles corresponding to the question.  
- **SQL**: The gold-standard SQL query for the question; executing this query yields the answer to the question.  
- **markdown_answer**: Contains the answer in markdown format.  
- **SQL_result**: The result obtained by executing the SQL query.  
- **answer**: The final answer in natural language format.

Using the **slots** labels, we can extract keywords from the **query**. Below is a sample code snippet for this process:
```
# Convert slots into keywords
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
            keyword_pair.append(f'企业名称:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'年份:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'月份:{keyword[0]}')
        elif keyword[-1] == 'land':
            keyword_pair.append(f'地块名称:{keyword[0]}')

    return keyword_pair

query = "请问广州市白云区万科未来森林小区的建筑密度是多少？"
slots = "O O B-city I-city I-city B-district I-district I-district B-community I-community I-community I-community I-community I-community O O O O O O O O O O O"
print(restore_keywords_from_query(query, slots))
```
output:
```
['城市:广州市', '区域:白云区', '项目名称:万科未来森林']
```
