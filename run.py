from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from datasets import load_dataset
import pdb




# add the main function here
if __name__ == '__main__':

	tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer", max_len=512, truncation = True)

	dataset = load_dataset("amazon_reviews_multi", "en")
	vocab_size = 20000
	config = RobertaConfig(
		vocab_size=vocab_size+5,
		max_position_embeddings=514,
		num_attention_heads=12,
		num_hidden_layers=12,
		type_vocab_size=1,
	)



	model = RobertaForMaskedLM(config=config)

	model.num_parameters()

	def tokenize_function(examples):
		return tokenizer(examples["review_body"], truncation = True)

	tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["review_body"])
	block_size = tokenizer.model_max_length