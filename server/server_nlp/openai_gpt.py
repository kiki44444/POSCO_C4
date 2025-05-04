import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
# import pickle
from transformers import GPT2TokenizerFast

DATASET_PATH = 'docent_sections.csv'
COMPLETIONS_MODEL = "text-davinci-002"
MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"
MAX_SECTION_LEN = 1200
SEPARATOR = "\n* "
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.5,
    "max_tokens": 400,
    "model": COMPLETIONS_MODEL,}
separator_len = 0


def get_doc_embedding(text:str):
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)


def get_query_embedding(text: str):
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)


def compute_doc_embeddings(df: pd.DataFrame):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()}


def load_embeddings(fname: str):
    """
    Read the document embeddings and their keys from a CSV.
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {(r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()}


def vector_similarity(x, y):
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()], reverse=True)
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        document_embeddings,
        show_prompt: bool = False):
    prompt = construct_prompt(
        query,
        document_embeddings,
        df)
    if show_prompt:
        print(prompt)
    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS)
    return response["choices"][0]["text"].strip(" \n")


def gpt_ready(api_key):
    global separator_len

    openai.api_key = api_key
    df = pd.read_csv(DATASET_PATH)
    df = df.set_index(["title", "heading"])
    context_embeddings = compute_doc_embeddings(df)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))
    # f"Context separator contains {separator_len} tokens"
    return df, context_embeddings

def gpt_qa(query, df, context_embeddings):
    answer = answer_query_with_context(query, df, context_embeddings)
    print(f"\nQ: {query}\nA: {answer}")
    return answer


if __name__ == '__main__':
    import sys
    argument = sys.argv
    api_key = argument[1]
    query = argument[2]

    df, embeddings = gpt_ready(api_key)
    gpt_qa(query, df, embeddings)