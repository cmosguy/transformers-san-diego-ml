#%% -*- coding: utf-8 -*-
#Intro to Transformers(Part 2 - Data + Pretraining)v2.ipynb
import os
from datasets import load_dataset


dataset = load_dataset("amazon_reviews_multi", "en")

dataset

roughly_word_size = 0
char_size = 0
for sample in dataset["train"]["review_body"]:
  roughly_word_size += len(sample.split())
  char_size += len(sample)
print(f"Characters in the set: {char_size}")
print(f"Roughly number of words in the set: {roughly_word_size}")


#%%

#So our small review dataset has about 6 million words. For context, book corpus has 800 million so we are a few orders of magnitude smaller. All of english wikipedia has about 2500 million. There are datasets like c4 that are cleaned up datasets from common crawl that have hundreds of gigabytes or even terabytes. https://huggingface.co/datasets/c4

#C4 was used to train the T5 model. This paper is very interesting for understanding pretraining and scaling across various axes. https://arxiv.org/abs/1910.10683

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
dir(models)

"""There are various options for tokenizers. Many models use WordPiece or Byte Pair Encoding. Here is a useful article explaining their differences. https://stackoverflow.com/questions/55382596/how-is-wordpiece-tokenization-helpful-to-effectively-deal-with-rare-words-proble/55416944#55416944

For this example we will use WordPiece just because it gives slightly nicer looking output
"""

#%%
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

vocab_size = 20000
tokenizer.train_from_iterator(dataset["train"]["review_body"], special_tokens=["","","",""], vocab_size = vocab_size)

#%%
os.makedirs('tokenizer', exist_ok=True)


#%%
tokenizer.save_model("tokenizer")
tokenizer.get_vocab()
tokenized_text = tokenizer.encode_batch(dataset["train"]["review_body"])

#%%
import numpy as np
tokenized_text[0]
np.array(tokenized_text[0].ids)
np.array(tokenized_text[0].tokens)
tokenizer.decode(tokenized_text[0].ids)

#%% !pip install transformers

#%%
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./tokenizer/vocab.json",
    "./tokenizer/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("", tokenizer.token_to_id("")),
    ("", tokenizer.token_to_id("")),
)
tokenizer.enable_truncation(max_length=512)

#%%
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=vocab_size+5,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

#%%
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512, truncation = True)

#%%
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

model.num_parameters()

def tokenize_function(examples):
    return tokenizer(examples["review_body"], truncation = True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["review_body"])
block_size = tokenizer.model_max_length
#%%

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = tokenized_datasets.remove_columns(['review_id', 'product_id', 'reviewer_id', 'stars', 'review_title', 'language', 'product_category'])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

#%%
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    f"amazon_reviews",
    evaluation_strategy = "steps",
    eval_steps = 500,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size = 16,
    # push_to_hub=True,
    fp16=True,
    num_train_epochs = 20

)

#%%
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm = True, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

#%%
import torch
torch.cuda.empty_cache()

trainer.train()

input = tokenizer("The dog chased the red <mask>", return_tensors = "pt")

input = input.to("cuda")

output = model(**input)

tokenizer.mask_token_id

input["input_ids"].to("cpu")

np.argwhere(input["input_ids"].to("cpu") == tokenizer.mask_token_id)[1]

mask_token_index = np.argwhere(input["input_ids"].to("cpu") == tokenizer.mask_token_id)[1]

mask_token_logits = output["logits"].to("cpu")[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
# We negate the array before argsort to get the largest, not the smallest, logits
top_5_tokens = np.argsort(-mask_token_logits.detach().cpu()[0])[:5].tolist()
print(mask_token_logits)
for token in top_5_tokens:
    print(f" {tokenizer.decode(token)}")

