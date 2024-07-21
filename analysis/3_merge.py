import json
import os

# 原始数据文件路径
data_file_path = '../tmp/crosscodeeval_testrun/detailed_results.json'
# 新的JSON文件路径，包含task_id列表
new_jsonl_file_path = '../data/python/line_completion_rg1_unixcoder_cosine_sim.jsonl'

# 用于存储分类后的数据
categorized_data = {}

# 读取原始数据并分类
with open(data_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        task_id = data['task_id']
        es_score = data['es']
        # 分类到步长为10的区间，确保不超过100
        classification_bin = (int(min(es_score, 100) // 10) * 10)
        if classification_bin not in categorized_data:
            categorized_data[classification_bin] = []
        categorized_data[classification_bin].append(task_id)

# 读取JSONL文件中的完整行
with open(new_jsonl_file_path, 'r') as new_file:
    new_data_lines = [json.loads(line) for line in new_file]

# 创建保存匹配结果的文件夹
output_dir = 'matched_task_ids_by_es_score'
os.makedirs(output_dir, exist_ok=True)

# 匹配task_id并将结果保存到对应的文件夹
for classification_bin, task_ids in categorized_data.items():
    # 创建以ES分数区间命名的文件夹
    bin_output_dir = os.path.join(output_dir, f'{classification_bin}-{min(classification_bin + 10, 100)}')
    os.makedirs(bin_output_dir, exist_ok=True)

    # 匹配task_id并将匹配的行保存到新文件
    matched_lines = [line for line in new_data_lines if line['metadata']['task_id'] in task_ids]
    output_file_path = os.path.join(bin_output_dir, 'matched_data.jsonl')

    # 写入匹配的行到JSONL文件
    with open(output_file_path, 'w') as output_file:
        for line in matched_lines:
            json.dump(line, output_file)
            output_file.write('\n')  # 写入换行符以保持JSONL格式

print(f"所有匹配的测试用例已按ES分数区间保存到 '{output_dir}' 文件夹中。")