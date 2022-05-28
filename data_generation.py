# generate fake data to improve robustness
from transformers import pipeline, set_seed
import torch
import random
import pickle
from tqdm import tqdm


NUMBERT_OF_SENTENCE = 5000

generator = pipeline('text-generation', model='gpt2', device=0, pad_token_id=50256) 

def get_sentence():
    #給定兩個prompt, 分別是user id 以及 第一個字
    prompt_1 = "@{}".format(random.randint(10000, 99999))
    prompt_2 = generator("", max_length=3)[0]['generated_text']
    template = "{} {}".format(prompt_1, prompt_2)

    return generator(template, max_length=30)[0]['generated_text']
data = []
for i in tqdm(range(NUMBERT_OF_SENTENCE)):
    data.append(get_sentence())

with open('./data/gpt2_data.pickle', 'wb') as f:
    pickle.dump(data,f)