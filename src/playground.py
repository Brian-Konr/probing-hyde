import requests

url = "http://localhost:8000/generate"

article = """
Federal prosecutors laid out their most extensive case to date against former President Donald Trump for his effort to overturn the 2020 election in a sweeping legal brief that was unsealed Wednesday by a federal judge who is weighing the explosive criminal charges against him.

The 165-page document, which lands weeks before an election in which Trump is taking another shot at the White House, offers new detail about special counsel Jack Smith’s investigation into the former president’s efforts to lean on state officials and paint a narrative of widespread fraud that prosecutors say Trump knew was untrue.

It includes new details of Trump’s frayed relationship with former Vice President Mike Pence; FBI evidence of Trump’s phone usage on January 6, 2021, when rioters overtook the US Capitol; and conversations with family members and others where the then-president was fighting his loss to Joe Biden.

Broadly, and in response to the Supreme Court’s ruling this summer that granted Trump sweeping immunity for official actions, Smith’s motion claims the former president took the steps he did as a political candidate – not as a president – and that, therefore, he is not entitled to protection from prosecution the justices identified in July.

“When the defendant lost the 2020 presidential election, he resorted to crimes to try to stay in office,” Smith wrote in the brief, which US District Judge Tanya Chutkan released in partially redacted form.
"""

payload = f"Summarize the following news: {article}"
response = requests.post(url, json={"message": payload, "gen_config": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "num_return_sequences": 2,
    "do_sample": True
}}).json()

print(response)
gen_texts = []
for gen in response:
    gen_texts.append(gen["generated_text"][-1]['content'])

# print(gen_texts)