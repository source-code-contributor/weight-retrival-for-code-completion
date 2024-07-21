import json
import os

import openai
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_source_file(source_file_path):
    prompt_list = []
    repositories = []
    filename_list = []
    source_data = []

    with open(source_file_path, 'r') as source_file:
        for line in source_file:
            data = json.loads(line.strip())

            source_data.append(data)

            query = data['prompt']
            repository = data['metadata']['repository']
            filename = data['metadata']['file']

            prompt_list.append(query)
            repositories.append(repository)
            filename_list.append(filename)

    return prompt_list, repositories, filename_list, source_data


import numpy as np
from scipy.spatial.distance import cosine

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def truncate_text_for_query(text, max_tokens=8100):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
        truncated_text = encoding.decode(tokens)
        return truncated_text
    else:
        return text

client = openai.OpenAI(api_key="key")

def get_embedding_openai(text, model):
    truncated_text = truncate_text_for_query(text)
    return client.embeddings.create(input=truncated_text, model=model).data[0].embedding

def cosine_similarity_score(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def retrieve_similar_chunks(source_queries, source_repositories, source_filename_list, retrieval_folder, N,
                            retrieval_model):
    all_similar_chunks = []

    for query, source_repository, source_filename in tqdm(
            zip(source_queries, source_repositories, source_filename_list), total=len(source_queries),
            desc="Retrieving similar chunks"):
        matching_filename = os.path.join(retrieval_folder, source_repository.replace('/', '_') + '.jsonl')
        if not os.path.exists(matching_filename):
            print("Error repository_filename: ", matching_filename)
            continue

        chunked_lists = []
        with open(matching_filename, 'r') as retrieval_file:
            for line in retrieval_file:
                data = json.loads(line.strip())
                chunked_file_name = data['filename']
                if chunked_file_name == source_filename:
                    continue

                for chunk, embedding in zip(data['chunked_list'], data['chunked_embedding_list']):
                    truncated_chunk = chunk

                    chunked_lists.append({
                        "original_chunk": chunk,
                        "truncated_chunk": truncated_chunk,
                        "filename": chunked_file_name,
                        "embedding": np.array(embedding)
                    })

        query_embedding = get_embedding_openai(query, retrieval_model)

        scores = []
        for chunk_info in chunked_lists:
            chunk_embedding = chunk_info["embedding"]
            similarity = cosine_similarity_score(query_embedding, chunk_embedding)
            scores.append(similarity)

        max_score_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]

        query_similar_chunks = []
        for idx in max_score_indices:
            similar_chunk = chunked_lists[idx]
            query_similar_chunks.append({
                "retrieved_chunk": similar_chunk["original_chunk"],
                "filename": similar_chunk["filename"],
                "score": float(scores[idx])
            })

        all_similar_chunks.append(query_similar_chunks)

    return all_similar_chunks


def integrate_and_write_output(source_data, similar_chunks, output_file_path):
    with open(output_file_path, 'w') as output_file:
        for data, chunks in zip(source_data, similar_chunks):

            text = "# Here are some relevant code fragments from other files of the repo:\n"
            for chunk in chunks:
                text += "# the below code fragment can be found in:\n"
                text += "# " + chunk["filename"] + "\n"
                text += "\n".join("# " + line for line in chunk["retrieved_chunk"].split('\n')) + "\n"

            crossfile_context = {
                "text": text,
                "list": chunks
            }

            new_data = data.copy()
            new_data["crossfile_context"] = crossfile_context

            output_file.write(json.dumps(new_data) + '\n')




def main():
    lang_list = ["typescript"]
    retrieval_model_list = ["text-embedding-3-large"]
    line_size_list = [10]

    for lang in lang_list:
        for retrieval_model in retrieval_model_list:
            for line_size in line_size_list:
                source_file_path = "../data/"+lang+"/line_completion_rg1_bm25.jsonl"  \
                    # Download the data from the following link: [Download Link](https://drive.google.com/file/d/17BjcCjYdzvN6-Ylr0AezC9m2jsvlvh4j/view?usp=sharing).
                    #After downloading, extract the files into the `\data` directory.


                retrieval_folder = "./chunk_folder/"+lang+"_"+retrieval_model+"_"+str(line_size)

                output_file_path = "../data/"+lang+"/"+lang+"_"+retrieval_model+"_"+str(line_size)+".jsonl"
                print(output_file_path)
                N = 100  # Top N similar chunks to retrieve

                source_queries, source_repositories, source_filename_list, source_data = read_source_file(source_file_path)

                similar_chunks = retrieve_similar_chunks(source_queries, source_repositories,
                                                         source_filename_list, retrieval_folder, N, retrieval_model)
                print(len(similar_chunks))

                integrate_and_write_output(source_data, similar_chunks, output_file_path)

if __name__ == "__main__":
    main()
