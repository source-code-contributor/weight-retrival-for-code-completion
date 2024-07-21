import os
import json

def chunk_code(file_path, chunk_size=10):
    with open(file_path, 'r', encoding='utf-8',errors='ignore') as f:
        lines = [line for line in f.readlines() if line.strip()]
        chunks = ['	'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    return chunks

def chunk_code_2(file_path, chunk_size=10, slide_size=10):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line for line in f.readlines() if line.strip()]
        chunks = []
        for i in range(0, len(lines) - chunk_size + 1, slide_size):
            chunk = '\t'.join(lines[i:i + chunk_size])
            chunks.append(chunk)
    return chunks



def save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size):
    def get_files_in_folder(folder_path, extension):
        files = []
        for root, _, filenames in os.walk(folder_path, followlinks=True):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files

    files = get_files_in_folder(repo_path, '.' + language)
    repo_name = os.path.basename(os.path.normpath(repo_path))
    output_path = os.path.join(chunk_folder, f'{repo_name}.jsonl')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file in files:
            try:
                chunks = chunk_code(file, chunk_size)
                relative_path = os.path.relpath(file, repo_path)
                filename = relative_path.replace(os.sep, '/')
                json.dump({'filename': filename, 'chunked_list': chunks}, outfile, ensure_ascii=False)
                outfile.write('\n')
            except FileNotFoundError as e:
                print(f"File not found: {file}. Skipping...")

def process_repositories(repo_folder, language, chunk_folder, chunk_size):
    for repo_name in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, repo_name)
        if os.path.isdir(repo_path):
            save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size)



for chunk_size in [10]:
    repo_folder = './sorted_repositories/typescript'
    language = 'ts'  # ['java', 'cs', 'ts', 'py']
    chunk_folder = './chunk_folder/typescript'+"_"+str(chunk_size)

    # Create the chunk folder if it doesn't exist
    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)

    # Process repositories
    process_repositories(repo_folder, language, chunk_folder, chunk_size)


