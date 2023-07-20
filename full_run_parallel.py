#%%
import torch
import re
import time as tic
import pandas as pd
import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration


#%%
# Our set of prompts, each designed to extract a single feature of interest
prompts = {
    "wfh": "Does this job state that it is remote working or work from home? Answer yes or no.  Job description: ",
    "sexism": "Rate how sexist this job description is on a scale of 1 (not sexist) to 5 (very sexist).  Job description: ",
    "company": "Tell me the name of the company advertising this job if it's present in the description.  Job description: ",
    "shop": "Does this job involve working in a retail shop?  Answer yes or no.  Job description: ",
    "tech": "Does this job involve working with databases?  Answer yes or no.  Job description: ",
    "grad": "Is this job description advertised at recent graduates or those with less experience?  Answer yes or no.  Job description: ",
    "sector": "What type of role is being advertised in this job description?  Reply with one of 'management', 'sales', 'administrative', 'technology/IT', 'logistics', or 'other'.  Job description:",
    "quals": "Does this job generally require a lot of training or a degree to perform? Answer yes or no.  Job description: ",
    "experience": "Does this job generally require experience to perform? Answer yes or no.  Job description: "
}

jobs_df = pd.read_csv("data/data job posts.csv", usecols=["jobpost"])

# Take a quick look at a few examples
for i in range(3):
    print(jobs_df['jobpost'].iloc[i][:300])
    print(" ----- ")


#%%
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl",
    resume_download=True,
    device_map="auto",      # Will automatically detect the GPU if everything's installed
    load_in_8bit=True)      # Loads weights as smaller, 4-bit representation (faster to run)

model.config.max_new_tokens = 100


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
            .input_ids.to("cuda")  # Tokenizer also needs to be told we're using GPU's now
    return input_ids


#%%
def _clean_output(raw):
    """ Helper, clean unhelpful tokens from returned text """
    clean = re.sub(r"(<pad>)|(</s>)", "", raw)
    return clean.strip()


def _do_prompt(prompt:str, texts:list):
    """ Helper, apply a specific prompt to one text """
    input_ids = prompt_tokenize(prompt, texts)
    output_tokens = model.generate(input_ids)
    output_text = tokenizer.decode(output_tokens[0])
    return _clean_output(output_text)


texts = jobs_df["jobpost"].to_list()
MAX_BATCH_SIZE = 60

t0 = tic.time()
for index, chunk in enumerate(np.array_split(texts, (len(texts) / MAX_BATCH_SIZE)+1)):
    
    # Set up something to collect results
    results = {}
    results["text"] = [re.sub(r"[^a-zA-Z0-9 ]", "", job)[:100]
                       for job in chunk]

    # Apply every prompt
    for key in prompts.keys():
        t1 = tic.time()
        torch.cuda.empty_cache()        # Clears any lingering memory
        results[key] = _do_prompt(prompts[key], chunk)
        t2 = tic.time()
        print(f"Time to respond on {key}: {int(t2 - t1)} seconds.")

    # And we'll save that
    pd.DataFrame(results).to_csv(f"example_{index}.csv", index=False)

print(f"Time for whole dataset: {int(t2 - t0)} seconds.")

#%%
import pandas as pd
import numpy as np

file_index = 0
file_pile = []
while file_index is not None:
    try:
        tmp = pd.read_csv(f"example_{file_index}.csv")
        file_pile.append(tmp)
        file_index += 1
    except:
        print(f"Ended on {file_index}")
        file_index = None
    
df = pd.concat(file_pile)

df.to_csv("example_complete.csv", index=False)

# %%
df = df.replace("yes", 1)
df = df.replace("no", 0)

df[['sector', 'grad', 'shop', 'tech', 'quals']].groupby("sector").aggregate("sum")

totals = df['sector'].value_counts()
percents = np.round(100.0 * totals / totals.sum(), 2)
quals = df[['sector', 'experience']].groupby('sector').agg(["sum", "count"])['experience']
quals["% needs qualification"] = np.round(100.0 * quals['sum'] / quals['count'], 2)

tabulated = pd.DataFrame({"Total": totals, "%": percents})
tabulated = tabulated.join(quals[["% needs qualification"]])

# db_jobs = df['tech'].value_counts()

tabulated

# %%
