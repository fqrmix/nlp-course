import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer)

model_name = "sberbank-ai/rugpt3large_based_on_gpt2"

sber_model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

anecdote_train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='/Users/fqrmix/Developer/other-projects/nlp-mezotaken/repositories/nlp-course/anecdote.txt',
    block_size=128)

training_args = TrainingArguments(
    overwrite_output_dir = True,
    output_dir = 'anecdote',
    per_device_train_batch_size = 1,
    learning_rate = 1e-7,
    num_train_epochs = 2,
    use_mps_device=False
)

trainer = Trainer(
    model = sber_model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = anecdote_train_dataset
)

trainer.train()
trainer.save_model()
