import json
import re
from typing import Dict, List, Tuple, Optional

def find_answer_span(context: str, answer: str) -> Optional[Tuple[int, int]]:
    """
    Find the span of the answer in the context.
    Returns tuple of (start_index, end_index) if found, None if not found.
    """
    start_idx = context.find(answer)
    if start_idx == -1:
        return None
    return (start_idx, start_idx + len(answer))

def verify_and_correct_example(context: str, answer: Dict) -> Dict:
    """
    Verify if the answer text exists in context and correct start/end positions.
    Returns updated answer dict with corrected positions or error flags.
    """
    answer_text = answer['text']
    span = find_answer_span(context, answer_text)
    
    result = {
        'text': answer_text,
        'original_start': answer.get('answer_start', -1),
        'is_valid': False,
        'error': None
    }
    
    if span is None:
        result['error'] = 'Answer text not found in context'
        return result
    
    result['is_valid'] = True
    result['corrected_start'] = span[0]
    result['corrected_end'] = span[1]
    return result

def contains_english(text: str) -> bool:
    """
    Check if the given text contains any English characters.
    """
    return bool(re.search(r'[a-zA-Z]', text))

def clean_squad_dataset(data: Dict) -> Tuple[Dict, List[Dict]]:
    """
    Process entire SQuAD dataset and correct answer spans.
    Filters out questions or answers containing English characters.
    Returns cleaned dataset and list of error reports.
    """
    cleaned_data = {'version': data.get('version', '1.1'), 'data': []}
    error_reports = []
    
    for article in data['data']:
        cleaned_article = {'title': article['title'], 'paragraphs': []}
        
        for para in article['paragraphs']:
            context = para['context']
            cleaned_qas = []
            
            for qa in para['qas']:
                question = qa['question']
                answer = qa['answers'][0]  # SQuAD 1.1 assumes single answer per question
                
                # Skip questions or answers with English characters
                if contains_english(question) or contains_english(answer['text']):
                    error_reports.append({
                        'id': qa['id'],
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'error': 'Contains English characters'
                    })
                    continue
                
                verification = verify_and_correct_example(context, answer)
                
                if verification['is_valid']:
                    cleaned_qa = {
                        'id': qa['id'],
                        'question': question,
                        'answers': [{
                            'text': verification['text'],
                            'answer_start': verification['corrected_start']
                        }]
                    }
                    cleaned_qas.append(cleaned_qa)
                else:
                    error_reports.append({
                        'id': qa['id'],
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'error': verification['error']
                    })
            
            if cleaned_qas:  # Only include paragraphs with valid QAs
                cleaned_article['paragraphs'].append({
                    'context': context,
                    'qas': cleaned_qas
                })
        
        if cleaned_article['paragraphs']:  # Only include articles with valid paragraphs
            cleaned_data['data'].append(cleaned_article)
    
    return cleaned_data, error_reports

def save_cleaned_dataset(cleaned_data: Dict, output_path: str):
    """Save the cleaned dataset to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

def save_error_report(error_reports: List[Dict], output_path: str):
    """Save the error reports to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(error_reports, f, ensure_ascii=False, indent=2)

# Example usage:
if __name__ == "__main__":
    # Load paths from JSON configuration
    config_path = 'D:/Desktop/FY/FYP/QA_Translator/config.json'  # Update this with the actual path to your config file
    with open(config_path, 'r', encoding='utf-8') as f:
        paths = json.load(f)
    
    input_file = paths['output_file']
    cleaned_output_file = paths['cleaned_output_file']
    error_output_file = paths['error_output_file']
    
    # Load the translated dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean the dataset
    cleaned_data, error_reports = clean_squad_dataset(data)
    
    # Save the cleaned dataset and error report
    save_cleaned_dataset(cleaned_data, cleaned_output_file)
    save_error_report(error_reports, error_output_file)
    
    # Print summary
    total_original_qas = sum(len(p['qas']) for a in data['data'] for p in a['paragraphs'])
    total_cleaned_qas = sum(len(p['qas']) for a in cleaned_data['data'] for p in a['paragraphs'])
    error_count = len(error_reports)
    
    print(f"Original QA pairs: {total_original_qas}")
    print(f"Cleaned QA pairs: {total_cleaned_qas}")
    print(f"Errors found: {error_count}")
    print(f"Cleaned dataset saved to: {cleaned_output_file}")
    print(f"Error report saved to: {error_output_file}")
