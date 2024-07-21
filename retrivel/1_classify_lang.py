import json
import os
import shutil




def process_jsonl_and_copy_folders(jsonl_file_path, source_folder_root, target_root_folder):
    unique_repositories = set()

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            repository = data['metadata']['repository']
            unique_repositories.add(repository)

    for repository in unique_repositories:
        source_folder = os.path.join(source_folder_root, repository)
        target_folder = os.path.join(target_root_folder, repository)

        if os.path.exists(source_folder):
            shutil.copytree(source_folder, target_folder, dirs_exist_ok=True, symlinks=True)
        else:
            print(f"Repository folder not found: {source_folder}")



# When running this code,
# first download the repositories for all programming languages from: https://drive.google.com/file/d/1ni-TFCszlfQB3-bLVQZjBi5z9tw4kZXe/view?usp=sharing
# Then extract and place them into crosscodeeval_rawdata_v1.1.
if __name__ == '__main__':

    for lang in ["typescript"]:
        jsonl_file_path = '../data/' + lang + '/line_completion.jsonl'
        source_folder_root = './crosscodeeval_rawdata_v1.1/crosscodeeval_rawdata'
        target_root_folder = './sorted_repositories/' + lang

        os.makedirs(target_root_folder, exist_ok=True)

        process_jsonl_and_copy_folders(jsonl_file_path, source_folder_root, target_root_folder)