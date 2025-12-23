import json
import logging
import os
import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, TrainingArguments, Trainer, EarlyStoppingCallback

# Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for faster matrix operations on Ampere GPUs (A6000)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read Train Data
with open(os.path.join(PROJECT_ROOT, "data/train.json"), "r", encoding="utf8") as read_file:
    train = json.load(read_file)

# Read Test Data
with open(os.path.join(PROJECT_ROOT, "data/test.json"), "r", encoding="utf8") as read_file:
    test = json.load(read_file)


def flatten_qa_dataset(dataset):
    contexts = []
    questions = []
    answers = []

    for group in dataset:          # group is a list
        for doc in group:          # doc is a dict
            context = doc["context"]
            for qa in doc["questions"]:
                contexts.append(context)
                questions.append(qa["question"])
                answers.append({
                    "text": qa["answer"]["text"],
                    "answer_start": qa["answer"]["answer_start"]
                })

    return {
        "context": contexts,
        "question": questions,
        "answers": answers
    }


train_data = flatten_qa_dataset(train)
test_data  = flatten_qa_dataset(test)

train_dataset = Dataset.from_dict(train_data)
eval_dataset  = Dataset.from_dict(test_data)


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

def preprocess_qa(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        sequence_ids = tokenized.sequence_ids(i)

        start_token = end_token = 0
        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != 1:
                continue
            if offset[0] <= start_char < offset[1]:
                start_token = idx
            if offset[0] < end_char <= offset[1]:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    tokenized.pop("offset_mapping")
    return tokenized

train_dataset = train_dataset.map(preprocess_qa, batched=True)
eval_dataset  = eval_dataset.map(preprocess_qa, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

eval_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)


model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased")

training_args = TrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "checkpoints/distilbert"),
    overwrite_output_dir=True,
    num_train_epochs=10,              # Reduced from 25 (early stopping will handle it)
    per_device_train_batch_size=128,  # DistilBERT is smaller, can use same or larger batch
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,               # Slightly lower LR for stability
    weight_decay=0.01,                # Regularization to prevent overfitting
    warmup_ratio=0.1,                 # Warmup for 10% of training
    eval_strategy="steps",
    eval_steps=200,                   # Evaluate more frequently
    logging_steps=50,
    save_strategy="steps",            # Save checkpoints
    save_steps=200,                   # Save at each eval
    save_total_limit=3,               # Keep only best 3 checkpoints
    load_best_model_at_end=True,      # Load best model when done
    metric_for_best_model="eval_loss",
    greater_is_better=False,          # Lower eval_loss is better
    report_to="none",
    bf16=True,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    remove_unused_columns=False,
    optim="adamw_torch_fused",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 evals
)

trainer.train()

# Save the best model
trainer.save_model(os.path.join(PROJECT_ROOT, "models/distilbert"))
tokenizer.save_pretrained(os.path.join(PROJECT_ROOT, "models/distilbert"))
trainer.evaluate()
