import fitz  # PyMuPDF
import numpy as np
from parrotai import ParrotAPI
import time
import json
import os
from dotenv import load_dotenv

parrot = ParrotAPI()

load_dotenv()

username = os.getenv('PARROT_USERNAME')
password = os.getenv('PARROT_PASSWORD')

login_resp = parrot.login(username=username, password=password)

def read_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return texts

def create_embeddings(texts, model="gte-large"):
    embeddings = []
    for text in texts:
        resp = parrot.create_text_embedding(text=text, model=model)
        task_id = resp['data']['task_id']
        timeout = time.time() + 60 
        while True:
            if time.time() > timeout:
                break
            time.sleep(1)
            result = parrot.result_text_embedding(task_id=task_id)
            if result['data']['data']['status'] == 'COMPLETED':
                break

        response = result['data']['data']['response']
        if response == '' or response is None:
            print(f"The empty string from text: {text[:30]}...")
            continue
        else:
            embedding_list = json.loads(response)
            embedding = np.array(embedding_list, dtype=np.float32)
            embeddings.append(embedding)
    return embeddings

def cosine_similarity(a, b):
    a = a.flatten() 
    b = b.flatten() 
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def find_most_relevant_text(question_embedding, text_embeddings):
    similarities = [cosine_similarity(question_embedding, text_emb) for text_emb in text_embeddings]
    most_relevant_index = np.argmax(similarities)
    return most_relevant_index

pdf_path = 'Simple_Data_for_RAG.pdf'

texts = read_pdf_text(pdf_path)

text_embeddings = create_embeddings(texts)

user_question = "What is the capital of France?"

question_embedding_resp = parrot.create_text_embedding(text=user_question, model="gte-large")
task_id_embedding = question_embedding_resp['data']['task_id']

timeout = time.time() + 60 
while True:
    if time.time() > timeout:
        break
    time.sleep(1)
    result_embedding = parrot.result_text_embedding(task_id=task_id_embedding)
    if result_embedding['data']['data']['status'] == 'COMPLETED':
        break

response_embedding = result_embedding['data']['data']['response']

if response_embedding == '' or response_embedding is None:
    print("The empty string from question.")
else:
    embedding_list = json.loads(response_embedding)
    question_embedding = np.array(embedding_list, dtype=np.float32)


most_relevant_index = find_most_relevant_text(question_embedding, text_embeddings)
relevant_text = texts[most_relevant_index]

prompt = f"You are a contextual analysis expert. I have a question and relevant information for you to find the answer and rewrite it in a suitable style.\n\nQuestion: {user_question}\n\nRelevant Information: {relevant_text}\n\n Answer: \n\n I just need you to quote the answer from the Relevant Information. You don't need to include any additional content."
response = parrot.text_generation(
    messages=[{"role": "user", "content": prompt}],
    model="gemma-7b",
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)


if response["data"]["is_success"]:
    content = response["data"]["data"]["response"]
    print(content)  
else:
    print("Unable to generate response.")

