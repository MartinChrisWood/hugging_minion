# Snippet will run a proper analysis I can then plot things from
# Convert float 16 on CPU if possible with torch_dtype=torch.float16
#%%
import torch
import re
import time as tic

from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Note how tokenizers and models are loaded and configured separately here
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl",
    resume_download=True,
    device_map="auto",
    load_in_8bit=True)
    # torch_dtype=torch.bfloat16)

# More tokens take more time to generate
model.config.max_new_tokens = 100


#%%
t1 = tic.time()
input_text = "Does the following text describe a dog?  Answer yes or no:  'It has scales and wings.'"

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
text = tokenizer.decode(outputs[0])

print(text)

t2 = tic.time()
print(f"Time to respond: {int(t2 - t1)} seconds.")


#%%
import pandas as pd

jobs_df = pd.read_csv("data/data job posts.csv", usecols=["jobpost"])

# Take a quick look at a few examples
for i in range(3):
    print(jobs_df['jobpost'].iloc[i][:300])
    print(" ----- ")

#%%
# Our set of prompts, each designed to extract a single feature of interest
prompts = {
    "wfh": "Does this job state that it is remote working or work from home? Answer yes or no.  Job description: ",
    "sexism": "Rate how sexist this job description is on a scale of 1 (not sexist) to 5 (very sexist).  Job description: ",
    "company": "Tell me the name of the company advertising this job if it's present in the description.  Job description: ",
    "shop": "Does this job involve working in a retail shop?  Answer yes or no.  Job description: ",
    "tech": "Does this job involve working with databases?  Answer yes or no.  Job description: "
}

def prompt_tokenize(prompt:str, texts:iter, tokenizer=tokenizer, limit:int=480):
    """
    Helper, wraps creating the input tokens for the LLM.
    """
    input_texts = [(prompt + text)[:limit] + "  Answer:" for text in texts]
    input_ids = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors="pt")\
            .input_ids.to("cuda")
    return input_ids


#%%
from transformers import pipeline

# Instantiate a classification model, trained for this kind of work
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                      device_map="cuda:0")
# Define the labels to evaluate against this text and classify
candidate_labels = ['finance', 'technology', 'retail', 'health', 'internship']


#%%
def _clean_output(raw):
    # Helper, clean unhelpful tokens from returned text
    clean = re.sub(r"(<pad>)|(</s>)", "", raw)
    return clean.strip()

def _do_prompt(prompt, text):
    # Helper, apply a specific prompt to one text
    input_ids = prompt_tokenize(prompt, [text])
    output_tokens = model.generate(input_ids)
    # REF FIX IN BLOG
    output_text = tokenizer.decode(output_tokens[0])
    return _clean_output(output_text)

#%%
results = []
for i, job in tqdm(enumerate(jobs_df.sample(500)["jobpost"])):
    record = {"index": i}

    # Applies all the Large language Model prompts
    for key in prompts.keys():
        torch.cuda.empty_cache()  # Clears any lingering memory
        record[key] = _do_prompt(prompts[key], job)

    # Applies the classifier
    labels = classifier(job, candidate_labels)
    for j in range(len(candidate_labels)):
        record[candidate_labels[j]] = labels['scores'][j]
    print(i, record)
    results.append(record)

df = pd.DataFrame(results)
df.to_csv("example.csv")
