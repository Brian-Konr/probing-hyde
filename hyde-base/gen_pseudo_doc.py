import requests
from pyserini.search import get_topics, get_qrels
from tqdm import tqdm
import json
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--topic-name", type=str, default="dl19-passage")
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--api-url", type=str, default="http://localhost:8000/generate")
parser.add_argument("--use-few-shot", type=bool, default=False)
parser.add_argument("--few-shot-path", type=str, default="")

args = parser.parse_args()

topics = get_topics(args.topic_name)
url = args.api_url
gen_config = {
    "max_new_tokens": args.max_new_tokens,
    "do_sample": False
}
gen_pseudo_docs = {"gen_config": gen_config, "topics": []}


def load_few_shot_examples():
    with open(args.few_shot_path) as f:
        data = json.load(f)
    composed_prompt = ""
    for example in data:
        composed_prompt += f"Query: {example['query']}\n"
        composed_prompt += f"Passage: {example['passage']}\n\n"
    return composed_prompt

for qid in tqdm(topics, desc=f"Generating pseudo-docs for {args.topic_name}"):
    query = topics[qid]["title"]
    prompt = f"You will be given a query, and you need to generate a passage that answers the query. The passage should be informative and relevant to the query\n"
    if args.use_few_shot:
        prompt += "Here are some examples of queries and passages:\n"
        prompt += load_few_shot_examples()
    
    prompt += f"Please generate a passage that answers the following query:\n"
    prompt += f"Query: {query}\n"
    prompt += f"Passage:"

    response = requests.post(url, json={"message": prompt, "gen_config": gen_config}).json()
    
    passage = response["generated_text"]
    word_predictions = response["word_predictions"]
    gen_pseudo_docs["topics"].append({
        "qid": qid,
        "query": query,
        "passage": passage,
        "word_predictions": word_predictions
    })

directory = f"pseudo-docs/{args.topic_name}"
os.makedirs(directory, exist_ok=True)

with open(f"{directory}/1.json", "w") as f:
    json.dump(gen_pseudo_docs, f)

print("Done!")
