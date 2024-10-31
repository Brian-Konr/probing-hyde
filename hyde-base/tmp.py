from pyserini.search import get_topics, get_qrels
from tqdm import tqdm
import json

url = "http://localhost:8000/generate"
topic_name = "dl19-passage"

topics = get_topics(topic_name)

info = []
for qid in topics:
    entry = {
        "qid": qid,
        "query": topics[qid]['title']
    }
    info.append(entry)

# save the query id and query title dict
with open(f"{topic_name}_topic_info.json", "w") as f:
    json.dump(info, f)