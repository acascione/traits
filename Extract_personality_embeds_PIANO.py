import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

device = torch.device("cuda")  # Use the first available GPU
print(device)

import pandas as pd
import numpy as np
import torch
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig

id_target = 4
max_length = 4096

def divide_text_into_batches(text, batch_size= int(max_length * 5)):
    words = text.split()
    
    batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
    
    batches = [' '.join(batch) for batch in batches]
    
    return batches


from datasets import load_dataset

dataset = load_dataset("PwNzDust/user_full_text" )

targets = ['agreeableness', 'openness', 'conscientiousness', 'extraversion','neuroticism']

target = targets[id_target]
target
print(f'DEALING WIHT {target} TARGET')

# Create a DataFrame using the dataset's 'author' and 'full_text' columns from the 'train' split
df = pd.DataFrame({
    'author': dataset['train']['author'], 
    'full_text': dataset['train']['full_text']
})

author_full_text_join_cleaned = df



authors_batches = []
texts = []
y_values = []

for i, row in author_full_text_join_cleaned.iterrows():
    batches = divide_text_into_batches(row['full_text'])
    #target_values = [row[target] for x in range(len(batches))]
    authors_batches += [row['author'] for x in range(len(batches))]
    texts += batches
   # y_values += target_values
    
bins = [0, 20, 40, 60, 80, 101]  # Note: upper bound of the last bin is 101 to include 100
labels = [0, 1, 2, 3, 4]


len(authors_batches)

max_length = 4096

texts[0]

model = LongformerForSequenceClassification.from_pretrained(f'PwNzDust/{target}_model_30',
                                                           gradient_checkpointing=False,
                                                           attention_window = max_length,
                                                           token = 'token_auth'
                                                           ).to(device)

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = max_length)

# Tokenize and move input to GPU
embeddings = []

# Loop over texts, tokenize, and get embeddings
for i, text in enumerate(texts):
    if (i % 1000 == 0):
        print(i)
    # Tokenize and move input to GPU

     
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    
    # Run inference on GPU
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract penultimate layer embeddings
    penultimate_layer_hidden_states = outputs.hidden_states[-2]
    penultimate_embeddings = torch.mean(penultimate_layer_hidden_states, dim=1)
    
    # Move embeddings to CPU and convert to numpy
    penultimate_embeddings_np = penultimate_embeddings.detach().cpu().numpy()
    
    # Add the embeddings to the list
    embeddings.append(penultimate_embeddings_np)

# Convert the list of embeddings into a single numpy array
embeddings_np = np.vstack(embeddings)
print(embeddings_np.shape)

# Create a DataFrame with embeddings
dataframe = pd.DataFrame(embeddings_np)

# Ensure that the length of 'authors_batches' matches the embeddings
assert len(authors_batches) == dataframe.shape[0], "Mismatch between authors and embeddings!"

# Add the author column to the DataFrame
dataframe['author'] = authors_batches

# Save the embeddings DataFrame to Parquet
dataframe.to_parquet(f'{target}_embeddings.parquet', index=False)

print(f"Embeddings saved to {target}_embeddings.parquet")
