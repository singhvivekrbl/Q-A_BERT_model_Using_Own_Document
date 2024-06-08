
import json
from transformers import BertTokenizerFast

def load_custom_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_custom_data(data, tokenizer):
   
    document = data['document']
    questions = data['questions']
    examples = []
   

    for question in questions:
        tokenized_example = tokenizer(
            question, document,
            truncation="only_second", max_length=384,
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length"
        )
        # examples.append(tokenized_example)
   
        examples.append({
            'input_ids': tokenized_example['input_ids'],
            'attention_mask': tokenized_example['attention_mask']
        })
    return examples

if __name__ == "__main__":
    custom_data_path = 'C:\\Users\\singh\\OneDrive\\Desktop\\Q-A_model\\Question_Answer.json'
    custom_data = load_custom_data(custom_data_path)
    
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'  # Use a model fine-tuned for question answering
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenized_custom_data = preprocess_custom_data(custom_data, tokenizer)
    
    with open('tokenized_custom_data.json', 'w') as f:
        json.dump(tokenized_custom_data, f)
