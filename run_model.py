import pandas as pd
import torch
from transformers import BertTokenizer
import load_data_run as load_data
import sys, time, datetime, random, csv
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np

# python run_model.py '/Users/emt/PycharmProjects/expo/data/descriptive investigations - US_fake_titles_all.csv' US_fake_titles_all_predictions.csv /Users/emt/PycharmProjects/expo/models/finetune_trained_06-12-21/

# If there's a GPU available...

device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained(sys.argv[3])
#config = BertConfig.from_json_file('../models/bert_classifier_2epoch_256size/config.json')
tokenizer = BertTokenizer.from_pretrained(sys.argv[3])

#model.cuda()

#load comments and labels from the input tsv
comments, url_ids = load_data.get_data(sys.argv[1])

#encode inputs using BERT tokenizer
input_ids = []


for comment in comments:
    encoded_comment = tokenizer.encode(comment, add_special_tokens = True, max_length=512,pad_to_max_length=True)
    input_ids.append(encoded_comment)

#define attention masks: if 0 it's a PAD, set to 0; else set to 1
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)


batch_size = 256

# Create the DataLoader for our training set.
# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions , true_labels = [], []

# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()

  # Store predictions and true labels
  predictions.append(logits)

print('    DONE.')

# Combine the predictions for each batch into a single list of 0s and 1s.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

with open(sys.argv[2], mode='w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter = ',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
  csv_writer.writerow(['min_id', 'prediction'])
  for url_id, prediction in zip(url_ids, flat_predictions):
    csv_writer.writerow([url_id, prediction])
