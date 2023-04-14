#%%
import re
import torch

import numpy as np
import pandas as pd
import time as tic

from transformers import T5Tokenizer, T5ForConditionalGeneration


#%%
# Check that torch has access cuda for computation
print(torch.cuda.is_available())

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-large",
    resume_download=True,
    device_map="auto",
    torch_dtype=torch.float16)

#%%
# Print the full model config
for key in dir(model.config):
    print(key)


#%%

# More complex queries take longer to answer.
model.config.max_new_tokens = 500

t1 = tic.time() 
input_text = "Does the following text describe a dog?  Answer yes or no:  'It has scales and wings.'"

input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
text = tokenizer.decode(outputs[0])

print(text)

t2 = tic.time()
print(f"Time to respond: {int(t2 - t1)} seconds.")


#%%
jobs_df = pd.read_csv("data/data job posts.csv")

#%%
prompts = {
    "wfh": "Does this job state that it is remote working or work from home?  Answer yes or no: ",
    "sexism": "Rate how sexist this job description is on a scale of 1 (not sexist) to 5 (very sexist): ",
    "company": "Tell me the name of the company advertising this job if it's present in the description: ",
    "shop": "Does this job involve working in a retail shop?  Answer yes or no: ",
    "tech": "Does this job involve working with databases?  Answer yes or no: "
}

def prompt_tokenize(prompt:str, texts:iter, tokenizer=tokenizer, limit:int=512):
    """
    Helper, wraps creating the input tokens for the LLM, meant to be edited.
    """
    input_texts = [(prompt + text)[:limit] for text in texts]
    input_ids = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        return_tensors="pt")\
            .input_ids.to("cuda")
    return input_ids

for i, job in enumerate(jobs_df["jobpost"][:70]):
    input_ids = prompt_tokenize(prompts["company"], [job])
    output = model.generate(input_ids)
    print(i, " --- ", tokenizer.decode(output[0]))


# %%
# This is how it SHOULD be used
input_ids = prompt_tokenize(prompts["company"], jobs_df["jobpost"][:50])
outputs = model.generate(input_ids)
print(tokenizer.batch_decode(outputs))
# %%
