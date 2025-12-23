# evaluate model performance on legal QA tasks from bert_qa_best folder with test set
import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
import nltk

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Download required NLTK data to project folder
nltk_data_path = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.insert(0, nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)


def flatten_data(raw_data):
    """Flatten the nested data structure into a list of QA pairs."""
    flattened = []
    for group in raw_data:
        for item in group:
            context = item['context']
            for qa in item['questions']:
                question = qa['question']
                answer_text = qa['answer']['text']
                answer_start = qa['answer']['answer_start']
                answer_end = answer_start + len(answer_text)
                flattened.append({
                    'question': question,
                    'context': context,
                    'answer_text': answer_text,
                    'answer_start': answer_start,
                    'answer_end': answer_end
                })
    return flattened


class LegalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        context = item['context']
        answer_start_char = item['answer_start']
        answer_end_char = item['answer_end']
        
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Convert character positions to token positions
        offset_mapping = inputs['offset_mapping'].squeeze().tolist()
        
        start_position = 0
        end_position = 0
        
        # Find the token positions for the answer
        for i, (start, end) in enumerate(offset_mapping):
            if start <= answer_start_char < end:
                start_position = i
            if start < answer_end_char <= end:
                end_position = i
                break
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'start_position': torch.tensor(start_position, dtype=torch.long),
            'end_position': torch.tensor(end_position, dtype=torch.long)
        }


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def evaluate_model(model, dataloader, device, tokenizer, test_data):
    model.eval()
    all_start_preds = []
    all_end_preds = []
    all_start_labels = []
    all_end_labels = []
    all_pred_texts = []
    all_true_texts = []
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_position'].to(device)
            end_positions = batch['end_position'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_preds = torch.argmax(start_logits, dim=1).cpu().numpy()
            end_preds = torch.argmax(end_logits, dim=1).cpu().numpy()
            start_labels = start_positions.cpu().numpy()
            end_labels = end_positions.cpu().numpy()

            # Extract predicted answer texts
            for i in range(len(start_preds)):
                input_id = input_ids[i].cpu().numpy()
                pred_start = start_preds[i]
                pred_end = end_preds[i]
                
                # Ensure end >= start
                if pred_end < pred_start:
                    pred_end = pred_start
                
                # Extract predicted tokens and decode
                pred_tokens = input_id[pred_start:pred_end + 1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                all_pred_texts.append(pred_text)
                
                # Get true answer text from original data
                true_text = test_data[idx]['answer_text']
                all_true_texts.append(true_text)
                idx += 1

            all_start_preds.extend(start_preds)
            all_end_preds.extend(end_preds)
            all_start_labels.extend(start_labels)
            all_end_labels.extend(end_labels)

    # Position-based metrics
    start_accuracy = accuracy_score(all_start_labels, all_start_preds)
    end_accuracy = accuracy_score(all_end_labels, all_end_preds)
    start_f1 = f1_score(all_start_labels, all_start_preds, average='weighted')
    end_f1 = f1_score(all_end_labels, all_end_preds, average='weighted')

    # Text-based metrics
    print("\nComputing text-based metrics...")
    
    # 1. Exact Match
    exact_matches = sum(1 for pred, true in zip(all_pred_texts, all_true_texts) 
                       if pred.lower().strip() == true.lower().strip())
    exact_match_score = exact_matches / len(all_pred_texts)
    
    # 2. BLEU Score
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    for pred, true in tqdm(zip(all_pred_texts, all_true_texts), desc="Computing BLEU", total=len(all_pred_texts)):
        reference = [nltk.word_tokenize(true.lower())]
        hypothesis = nltk.word_tokenize(pred.lower())
        if len(hypothesis) == 0:
            bleu_scores.append(0.0)
        else:
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
            bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores)
    
    # 3. BERT Score
    print("Computing BERT Score (this may take a while)...")
    P, R, F1 = bert_score(all_pred_texts, all_true_texts, lang="en", verbose=True)
    avg_bert_precision = P.mean().item()
    avg_bert_recall = R.mean().item()
    avg_bert_f1 = F1.mean().item()
    
    # 4. Cosine Similarity using Sentence Transformers
    print("Computing Cosine Similarity...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    pred_embeddings = sentence_model.encode(all_pred_texts, show_progress_bar=True)
    true_embeddings = sentence_model.encode(all_true_texts, show_progress_bar=True)
    
    cosine_scores = []
    for pred_emb, true_emb in zip(pred_embeddings, true_embeddings):
        cos_sim = cosine_similarity([pred_emb], [true_emb])[0][0]
        cosine_scores.append(cos_sim)
    avg_cosine_similarity = np.mean(cosine_scores)

    # Build predictions list with questions, contexts, predictions and ground truth
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            'question': item['question'],
            'context': item['context'],
            'ground_truth': all_true_texts[i],
            'prediction': all_pred_texts[i],
            'exact_match': all_pred_texts[i].lower().strip() == all_true_texts[i].lower().strip()
        })

    return {
        'start_accuracy': float(start_accuracy),
        'end_accuracy': float(end_accuracy),
        'start_f1': float(start_f1),
        'end_f1': float(end_f1),
        'exact_match': float(exact_match_score),
        'bleu_score': float(avg_bleu),
        'bert_score_precision': float(avg_bert_precision),
        'bert_score_recall': float(avg_bert_recall),
        'bert_score_f1': float(avg_bert_f1),
        'cosine_similarity': float(avg_cosine_similarity),
        'num_samples': len(all_pred_texts),
        'predictions': predictions
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect model type using AutoTokenizer and AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    model.to(device)

    raw_data = load_data(args.test_file)
    test_data = flatten_data(raw_data)
    print(f"Loaded {len(test_data)} QA pairs from test set")
    
    test_dataset = LegalQADataset(test_data, tokenizer, max_length=args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    results = evaluate_model(model, test_dataloader, device, tokenizer, test_data)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"\nPosition-based Metrics:")
    print(f"  Start Position - Accuracy: {results['start_accuracy']:.4f}, F1 Score: {results['start_f1']:.4f}")
    print(f"  End Position   - Accuracy: {results['end_accuracy']:.4f}, F1 Score: {results['end_f1']:.4f}")
    print(f"\nText-based Metrics:")
    print(f"  Exact Match:        {results['exact_match']:.4f}")
    print(f"  BLEU Score:         {results['bleu_score']:.4f}")
    print(f"  BERT Score (P/R/F1): {results['bert_score_precision']:.4f} / {results['bert_score_recall']:.4f} / {results['bert_score_f1']:.4f}")
    print(f"  Cosine Similarity:  {results['cosine_similarity']:.4f}")
    print(f"\nTotal Samples Evaluated: {results['num_samples']}")
    print("="*60)

    # Save results to file
    output_file = args.output_file if args.output_file else 'evaluation_results.json'
    
    # Separate predictions from metrics for saving
    predictions = results.pop('predictions')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {output_file}")
    
    # Save predictions to a separate file
    predictions_file = output_file.replace('.json', '_predictions.json')
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Legal QA Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model directory')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test dataset file (JSON format)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help='Path to save evaluation results')
    args = parser.parse_args()
    main(args)
