#分析不同性能之间，prompt和crossfile_context之间的区别

import json
import os

# 假定的文件夹路径，根据您的实际路径进行调整
base_dir = "./matched_task_ids_by_es_score"  # 请将此路径替换为您存放文件夹的实际路径

folder_names = [f"{i}-{i + 10}" for i in range(0, 111, 10)]

for folder_name in folder_names:
    folder_path = os.path.join(base_dir, folder_name)
    jsonl_file_path = os.path.join(folder_path, "matched_data.jsonl")  # 假定每个文件夹下的jsonl文件名是已知的

    try:
        with open(jsonl_file_path, 'r') as file:
            total_score = 0
            line_count = 0
            for line in file:
                data = json.loads(line)
                scores = [item['score'] for item in data['crossfile_context']['list']]
                line_average_score = sum(scores) / len(scores) if scores else 0
                total_score += line_average_score
                line_count += 1

            file_average_score = total_score / line_count if line_count else 0
            print(f"Folder {folder_name}: File average score = {file_average_score}")
    except FileNotFoundError:
        print(f"File {jsonl_file_path} not found.")