# Getting started with transformers from Huggingface

## Hardware Requirements

This in reference to [Flan-T5-XL](https://huggingface.co/google/flan-t5-xl):
- > 40GB free hard disk space for the model itself.
- ~ 20 GB RAM for 16-bit model, or 8GB or more VRAM if using said model on CUDA.
- An Nvidia GPU that supports CUDA if you want it to run at a reasonable speed.

## Software Requirements
- CUDA drivers for GPU computing. CUDA can be [downloaded from Nvidia's website](https://developer.nvidia.com/cuda-downloads).
- For windows:  Python 3.9.16 or earlier.  Pytorch does not yet support later python editions on windows (but it does on linux).  It's easiest to use [Anaconda]() for installing different python versions.
- PyTorch, OR Tensorflow python packages.  I've found PyTorch easiest to install, using pip. Follow the instructions on the [PyTorch website]() for your specific system, rather than using the default pip torch package.
- Huggingface transformers package, installed with `pip install transformers` (or this project's `requirements.txt` file)

Technically Apple's new ARM processors support GPU computing through their Metal Performance Shaders (MPS).
In practice, shrinking a large language model requires converting its weights and inputs to 16 bit numbers (a new method goes smaller) and the 
conversions are not yet supported for at least one tokeniser - the one I need - on MPS and CPU.  I don't know if this is a widespread issue.

`pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu117`


## Decoding Strategies

https://huggingface.co/docs/transformers/generation_strategies

Searching for a most probable output is essentially a tree-search problem.
The method for doing so and finding a reasonable text output given some
input is therefore customisable.  The page linked above goes into different
strategies for doing this.


## Notes on Prompt Engineering

I'm only using Flan-T5-XL, due to my poor unfortunate budget graphics card.

### Yes or no
In trying to get it to tell me if a job involved working in a high street shop I found that:

`"Does this job involve working in a shop?  Answer yes or no: "`

Works, but also picks up a job working in a machine shop (index 56).  A more specific version:

`"Does this job involve working in a retail shop?  Answer yes or no: "`

Works just fine.

### Likert scale
In trying to get it to rate job ads on a scale of 1 to 5 for sexism I found that:

`"Rate how sexist this job description is on a scale of 1 (not sexist) to 5 (very sexist): "`

Works, but:

`"Rate how sexist this job description is on a scale of 1 to 5: "`

Does not work, in detecting record index three, which requests a young woman with a "Nice-looking exterior" (Jeez!).  I suspect that this is a harder Q for a transformer model because it needs to turn a concept into a number.

### Information extraction
In trying to get it to give me the name of the company advertising the job I found that:

`"Tell me the name of the company advertising this job if it's present in the description: "`

Works, but:

`"Tell me the name of the company advertising this job: "`

Only mostly works.  In the latter case, where the name was not obvious the machine just hallucinated a bunch of job-related-sounding text.

### Asking for a justification

Currently the model stops once it reaches the end-of-sentence token.  An unfortunate side effect is that if you ask it `"Answer X yes or no.  Explain why: "` it stops generating text after the "yes" or "no" part.  This is a
consequence of the generator settings so it can be overridden, I just don't know how yet.
