import json
import os

# 假设数据存储在'data.json'文件中
data_file = '../tmp/crosscodeeval_testrun/detailed_results.json'

# 用于存储分类后的数据
categorized_data = {}

# 读取文件并处理数据
with open(data_file, 'r') as file:
    for line in file:
        # 解析每行的JSON数据
        data = json.loads(line)
        task_id = data['task_id']
        es_score = data['es']

        # 根据es分数进行分类，步长为10
        classification_bin = es_score // 10 * 10

        # 创建一个包含步长的范围字符串
        classification_bin_range = f"{classification_bin}-{classification_bin + 10}"

        # 将task_id添加到对应的分类中
        if classification_bin_range not in categorized_data:
            categorized_data[classification_bin_range] = []
        categorized_data[classification_bin_range].append(task_id)

# 打印分类后的数据
for bin_range, task_ids in categorized_data.items():
    print(f"ES Score Range: {bin_range}")
    for task_id in task_ids:
        print(f" - {task_id}")
    # print(len(task_ids))
    print()