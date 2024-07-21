import json

# 假设task_id列表存储在一个名为'task_ids.txt'的文本文件中，每行一个task_id
task_ids_file = 'task_ids.txt'
# 假设包含测试用例的JSON文件名为'test_cases.json'
input_json_file = 'test_cases.json'
# 输出文件名
output_json_file = 'matched_test_cases.json'

# 读取task_id列表
with open(task_ids_file, 'r') as f:
    task_ids = [line.strip() for line in f.readlines()]

# 读取JSON文件并解析为Python对象
with open(input_json_file, 'r') as f:
    test_cases = json.load(f)

# 创建一个新的列表来存储匹配的测试用例
matched_cases = []

# 遍历测试用例并检查task_id是否在给定的列表中
for case in test_cases:
    if case['metadata']['task_id'] in task_ids:
        matched_cases.append(case)

# 将匹配的测试用例写入新的JSON文件
with open(output_json_file, 'w') as f:
    json.dump(matched_cases, f, indent=2)

print(f"匹配的测试用例已保存到 '{output_json_file}'")