import os
import json
import openai

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def truncate_text_for_code(text, max_tokens=8100):
    text = text.replace("<|endoftext|>", "")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        # 将token截断到允许的最大数量
        tokens = tokens[:max_tokens]
        # 将tokens转回文本
        truncated_text = encoding.decode(tokens)
        return truncated_text
    else:
        return text

client = openai.OpenAI(api_key="key")

def get_embeddings(client, texts, model_name, batch_size=10):
    if len(texts) == 0:
        return []

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        truncated_texts = [truncate_text_for_code(text) for text in batch_texts]
        response = client.embeddings.create(input=truncated_texts, model=model_name)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings


def chunk_code(file_path, chunk_size=10):
    with open(file_path, 'r', encoding='utf-8',errors='ignore') as f:
        lines = [line for line in f.readlines() if line.strip()]
        chunks = ['	'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    return chunks

def get_files_in_folder(folder_path, extension):
    files = []
    for root, _, filenames in os.walk(folder_path, followlinks=True):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

from tqdm import tqdm
def save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size, model_name):
    files = get_files_in_folder(repo_path, '.' + language)
    repo_name = os.path.basename(os.path.normpath(repo_path))
    output_path = os.path.join(chunk_folder, f'{repo_name}.jsonl')

    if os.path.exists(output_path):
        print(f"Output file already exists for {repo_name}, skipping...")
        return

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for file in tqdm(files, desc="Processing files"):
            try:
                chunks = chunk_code(file, chunk_size)
                embeddings = get_embeddings(client, chunks, model_name)
                relative_path = os.path.relpath(file, repo_path)
                filename = relative_path.replace(os.sep, '/')
                json.dump({'filename': filename, 'chunked_list': chunks, 'chunked_embedding_list': embeddings}, outfile, ensure_ascii=False)
                outfile.write('\n')
            except FileNotFoundError:
                print(f"File not found: {file}. Skipping...")

def process_repositories(repo_folder, language, chunk_folder, chunk_size, model_name):
    for repo_name in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, repo_name)
        if os.path.isdir(repo_path):
            save_chunks_to_jsonl(repo_path, language, chunk_folder, chunk_size, model_name)

language = 'typescript'
language_type = 'ts'

for model_name in ["text-embedding-3-large"]:
    repo_folder = './sorted_repositories/'+ language
    chunk_size = 10
    chunk_folder = './chunk_folder/' + language + '_' + model_name + '_' + str(chunk_size)

    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)

    process_repositories(repo_folder, language_type, chunk_folder, chunk_size, model_name)
