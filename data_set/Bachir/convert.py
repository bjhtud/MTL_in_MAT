input_file = '/data/horse/ws/jibi984b-llm-2/llm_regression/data_set/Bachir/hu.csv'
output_file = '/data/horse/ws/jibi984b-llm-2/llm_regression/data_set/Bachir/hu.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有英文逗号为英文句号
content = content.replace(',', '.')

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("替换完成！")