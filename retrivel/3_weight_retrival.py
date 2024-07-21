import json
import os
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.cuda.amp import autocast
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import Counter
import math

# 语义模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_for_sematic = AutoTokenizer.from_pretrained("microsoft/unixcoder-base") # microsoft/unixcoder-base
model = AutoModel.from_pretrained("microsoft/unixcoder-base")


def bm25_retrieval(query, chunked_lists):
    tokenized_corpus = [tokenizer.tokenize(chunk["truncated_chunk"]) for chunk in chunked_lists]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenizer.tokenize(query))
    return scores

def calculate_frequency_threshold(corpus, percentile):
    total_freq = Counter()
    for document in corpus:
        total_freq.update(document)
    frequencies = list(total_freq.values())
    frequencies.sort()
    threshold_index = int(len(frequencies) * (percentile / 100))
    return frequencies[threshold_index]


def weight_retrieval(query, chunked_lists, weight_type):
    query_terms = tokenizer.tokenize(query)

    if weight_type == 'line':
        weights = {word: i+1 for i, word in enumerate(query_terms)}
    elif weight_type == 'e^x':
        weights = {word: math.exp((i + 1) / 30) for i, word in enumerate(query_terms)}
    elif weight_type == 'log':
        weights = {word: math.log(i + 1) for i, word in enumerate(query_terms)}
    elif weight_type == 'power':
        weights = {word: (i + 1) ** 1.1 for i, word in enumerate(query_terms)}
    else:
        raise ValueError("Unknown weight type")

    tokenized_corpus = [tokenizer.tokenize(chunk["truncated_chunk"]) for chunk in chunked_lists]
    bm25 = BM25Okapi(tokenized_corpus)

    def get_weighted_bm25_scores(query, weights, bm25, corpus, percentile=97):
        freq_threshold = calculate_frequency_threshold(corpus, percentile)

        total_freq = Counter()
        for document in corpus:
            total_freq.update(document)

        doc_scores = []
        for document in corpus:
            doc_dict = Counter(document)
            doc_len = len(document)
            score = 0
            for word in query:
                if word in bm25.idf:
                    term_freq = doc_dict[word]
                    idf = bm25.idf[word]
                    term_weight = weights.get(word, 1) if total_freq[word] < freq_threshold else 1
                    weighted_tf = (term_freq * (bm25.k1 + 1) * term_weight) / (
                                term_freq + bm25.k1 * (1 - bm25.b + bm25.b * (doc_len / bm25.avgdl)))
                    score += idf * weighted_tf
            doc_scores.append(score)
        return doc_scores

    scores = get_weighted_bm25_scores(query_terms, weights, bm25, tokenized_corpus)
    return scores

def bert_retrieval(query, chunked_lists, batch_size=64):
    # model = AutoModel.from_pretrained("microsoft/unixcoder-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    query_tokens1 = tokenizer_for_sematic.tokenize(query)[-512:]
    truncated_query = tokenizer_for_sematic.convert_tokens_to_string(query_tokens1)
    query_tokens = tokenizer_for_sematic(truncated_query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with autocast():
        with torch.no_grad():
            query_embedding = model(**query_tokens).last_hidden_state.mean(1)

    chunk_texts = [chunk["truncated_chunk"] for chunk in chunked_lists]
    chunk_encodings = tokenizer_for_sematic(chunk_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    chunk_dataset = TensorDataset(chunk_encodings['input_ids'].to(device), chunk_encodings['attention_mask'].to(device))
    chunk_dataloader = DataLoader(chunk_dataset, sampler=SequentialSampler(chunk_dataset), batch_size=batch_size)

    scores = []
    model.eval()
    with autocast():
        with torch.no_grad():
            for batch in chunk_dataloader:
                input_ids, attention_mask = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                chunk_embeddings = outputs.last_hidden_state.mean(1)
                batch_scores = cosine_similarity(query_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())
                scores.extend(batch_scores.flatten())

    model.to("cpu")

    return scores


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



def retrieve_similar_chunks(source_queries, source_repositories, source_filename_list, retrieval_folder, N, retrieval_model, w_type):
    all_similar_chunks = []

    for query, source_repository, source_filename in tqdm(zip(source_queries, source_repositories, source_filename_list), total=len(source_queries), desc="Retrieving similar chunks"):
        matching_filename = os.path.join(retrieval_folder, source_repository.replace('/', '_') + '.jsonl')
        if not os.path.exists(matching_filename):
            print("error repository_filename: ", matching_filename)

            continue

        chunked_lists = []
        with open(matching_filename, 'r') as retrieval_file:
            for line in retrieval_file:
                data = json.loads(line.strip())
                chunked_file_name = data['filename']
                if chunked_file_name == source_filename:
                    continue

                for chunk in data['chunked_list']:
                    truncated_chunk = chunk

                    chunked_lists.append({
                        "original_chunk": chunk,
                        "truncated_chunk": truncated_chunk,
                        "filename": chunked_file_name
                    })

        truncated_query = query

        if retrieval_model == 'BM25':
            scores = bm25_retrieval(truncated_query, chunked_lists)
        elif retrieval_model == 'weight_retrieval':
            scores = weight_retrieval(truncated_query, chunked_lists, w_type)
        elif retrieval_model == 'Semantic_Model':
            scores = bert_retrieval(truncated_query, chunked_lists)
        else:
            raise ValueError("Unsupported retrieval model. Choose either 'BM25' or 'BERT'.")

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
    retrieval_model_list = ["weight_retrieval"]
    line_size_list = [10]

    for lang in lang_list:
        for retrieval_model in retrieval_model_list:
            for line_size in line_size_list:
                for w_type in ['line', 'e^x', 'log', 'power']:
                    source_file_path = "../data/"+lang+"/line_completion_rg1_bm25.jsonl"
                        # Download the data from the following link: [Download Link](https://drive.google.com/file/d/17BjcCjYdzvN6-Ylr0AezC9m2jsvlvh4j/view?usp=sharing).
                        # After downloading, extract the files into the `\data` directory.

                    retrieval_folder = "./chunk_folder/" + lang + "_" + str(line_size)

                    output_file_path = "../data/"+lang+"/"+lang+"_"+retrieval_model+"_"+str(w_type)+".jsonl"
                    print(output_file_path)
                    N = 5  # Top N similar chunks to retrieve

                    source_queries, source_repositories, source_filename_list, source_data = read_source_file(source_file_path)

                    similar_chunks = retrieve_similar_chunks(source_queries, source_repositories,
                                                             source_filename_list, retrieval_folder, N, retrieval_model, w_type)
                    print(len(similar_chunks))

                    integrate_and_write_output(source_data, similar_chunks, output_file_path)

if __name__ == "__main__":
    main()
