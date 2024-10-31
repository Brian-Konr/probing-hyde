import requests
# from pyserini.search import get_topics, get_qrels
from tqdm import tqdm
import json

url = "http://localhost:8000/generate"
topic_name = "dl19-passage"

# topics = get_topics(topic_name)
with open("dl19-passage_topic_info.json", "r") as f:
    topics = json.load(f)

gen_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 1,
    "do_sample": True,
    "num_return_sequences": 8,
}

gen_pseudo_docs = {"gen_config": gen_config}


for topic in tqdm(topics, desc=f"Generating pseudo-docs for {topic_name}"):
    query = topic["query"]
    prompt = f"""Please write a passage to answer the question
Question: {query}
Passage:"""

    response = requests.post(url, json={"message": prompt, "gen_config": gen_config}).json()
    
    ans = [] # candidate passages
    for res in response:
        content = res["generated_text"][-1]["content"] # -1 indicates the last entry in generated_text, which is the assistant's response
        ans.append(content)
    topic["generated_passages"] = ans


gen_pseudo_docs["topics"] = topics
with open(f"{topic_name}_pseudo_docs_8rep.json", "w") as f:
    json.dump(gen_pseudo_docs, f)

print("Done!")
