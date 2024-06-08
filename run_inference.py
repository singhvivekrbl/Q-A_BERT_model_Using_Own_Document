import json
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

def get_answers(model, tokenizer, tokenized_data, original_data):
    model.eval()
    answers = []

    questions = original_data['questions']
    document = original_data['document']


    for i, tokenized_example in enumerate(tokenized_data):
        input_ids = torch.tensor(tokenized_example['input_ids'])  # Add batch dimension
        attention_mask = torch.tensor(tokenized_example['attention_mask'])  # Add batch dimension
        
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1

            answer = tokenizer.convert_tokens_to_string(all_tokens[answer_start:answer_end])
            answers.append({
                'question': questions[i],
                'context': document,
                'predicted_answer': answer
            })
    return answers

if __name__ == "__main__":
    custom_data_path = 'C:\\Users\\singh\\OneDrive\\Desktop\\Q-A_model\\Question_Answer.json'
    tokenized_data_path = 'tokenized_custom_data.json'
    
    with open(custom_data_path, 'r') as f:
        custom_data = json.load(f)
    
    with open(tokenized_data_path, 'r') as f:
        tokenized_custom_data = json.load(f)
    
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'  # Use a model fine-tuned for question answering
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    
    answers = get_answers(model, tokenizer, tokenized_custom_data, custom_data)
    
    for answer in answers:
        print(f"Question: {answer['question']}")
        print(f"Predicted Answer: {answer['predicted_answer']}")
        
