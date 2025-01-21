import json
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from tqdm import tqdm
import time
from datetime import datetime

def load_config(config_file):
    """Load configuration from a JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def batch_translate_texts(client, texts):
    """
    Translate texts in smaller batches to respect API limits.
    Maximum 128 segments per request.
    """
    if not texts:
        return []
        
    try:
        # Split into batches of 128 or fewer
        batch_size = 128
        all_translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = client.translate(
                batch,
                target_language='si',
                source_language='en'
            )
            translations = [result['translatedText'] for result in results]
            all_translations.extend(translations)
            time.sleep(0.1)  # Rate limiting between sub-batches
            
        return all_translations
    except Exception as e:
        print(f"\nBatch translation error: {e}")
        return texts

def translate_squad_batch(config_file):
    """
    Translates the SQuAD dataset using configurations from a JSON file.
    """
    try:
        # Load config
        config = load_config(config_file)
        input_file = config['input_file']
        output_file = config['output_file']
        credentials_path = config['credentials_path']
        max_contexts = config.get('max_contexts', 1000)
        batch_size = config.get('batch_size', 5)
        
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize translation client
        translate_client = translate.Client(credentials=credentials)
        
        # Load SQuAD dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)
        
        # Initialize output structure
        translated_data = {
            "version": squad_data.get("version", "1.1"),
            "data": []
        }
        
        # Stats tracking
        stats = {
            'contexts_processed': 0,
            'qa_pairs_processed': 0,
            'start_time': datetime.now()
        }
        
        # Process articles
        with tqdm(total=max_contexts) as pbar:
            for article in squad_data['data']:
                if stats['contexts_processed'] >= max_contexts:
                    break
                
                current_article = {
                    "title": article['title'],
                    "paragraphs": []
                }
                
                # Process paragraphs in smaller batches
                current_batch_contexts = []
                current_batch_paras = []  # Store full paragraph data
                
                for para in article['paragraphs']:
                    if stats['contexts_processed'] >= max_contexts:
                        break
                        
                    current_batch_contexts.append(para['context'])
                    current_batch_paras.append(para)
                    
                    # Process batch when it reaches batch_size or end of article
                    if len(current_batch_contexts) >= batch_size:
                        # Translate contexts
                        translated_contexts = batch_translate_texts(translate_client, current_batch_contexts)
                        
                        # Process each paragraph in the batch
                        for idx, (orig_para, trans_context) in enumerate(zip(current_batch_paras, translated_contexts)):
                            # Collect all questions and answers for this paragraph
                            qa_texts = []
                            for qa in orig_para['qas']:
                                qa_texts.append(qa['question'])
                                for answer in qa['answers']:
                                    qa_texts.append(answer['text'])
                            
                            # Translate all QA pairs
                            translated_qa_texts = batch_translate_texts(translate_client, qa_texts)
                            
                            # Create translated paragraph
                            translated_para = {
                                "context": trans_context,
                                "qas": []
                            }
                            
                            # Process QA pairs
                            qa_idx = 0
                            for qa in orig_para['qas']:
                                translated_qa = {
                                    "id": qa['id'],
                                    "question": translated_qa_texts[qa_idx],
                                    "answers": []
                                }
                                qa_idx += 1
                                
                                for answer in qa['answers']:
                                    translated_qa['answers'].append({
                                        "text": translated_qa_texts[qa_idx],
                                        "answer_start": answer['answer_start']
                                    })
                                    qa_idx += 1
                                
                                translated_para['qas'].append(translated_qa)
                                stats['qa_pairs_processed'] += 1
                            
                            current_article['paragraphs'].append(translated_para)
                            stats['contexts_processed'] += 1
                            pbar.update(1)
                            
                            # Print progress
                            if stats['contexts_processed'] % 10 == 0:
                                elapsed_time = (datetime.now() - stats['start_time']).total_seconds() / 60
                                print(f"\nProcessed: {stats['contexts_processed']}/{max_contexts} contexts")
                                print(f"Time elapsed: {elapsed_time:.1f} minutes")
                        
                        # Clear batches
                        current_batch_contexts = []
                        current_batch_paras = []
                
                # Process remaining paragraphs in the last batch
                if current_batch_contexts:
                    # Same processing as above for the final batch
                    translated_contexts = batch_translate_texts(translate_client, current_batch_contexts)
                    
                    for idx, (orig_para, trans_context) in enumerate(zip(current_batch_paras, translated_contexts)):
                        qa_texts = []
                        for qa in orig_para['qas']:
                            qa_texts.append(qa['question'])
                            for answer in qa['answers']:
                                qa_texts.append(answer['text'])
                        
                        translated_qa_texts = batch_translate_texts(translate_client, qa_texts)
                        
                        translated_para = {
                            "context": trans_context,
                            "qas": []
                        }
                        
                        qa_idx = 0
                        for qa in orig_para['qas']:
                            translated_qa = {
                                "id": qa['id'],
                                "question": translated_qa_texts[qa_idx],
                                "answers": []
                            }
                            qa_idx += 1
                            
                            for answer in qa['answers']:
                                translated_qa['answers'].append({
                                    "text": translated_qa_texts[qa_idx],
                                    "answer_start": answer['answer_start']
                                })
                                qa_idx += 1
                            
                            translated_para['qas'].append(translated_qa)
                            stats['qa_pairs_processed'] += 1
                        
                        current_article['paragraphs'].append(translated_para)
                        stats['contexts_processed'] += 1
                        pbar.update(1)
                
                # Add article to output
                if current_article['paragraphs']:
                    translated_data['data'].append(current_article)
                
                # Save progress regularly
                if stats['contexts_processed'] % 100 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        # Save final output
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        # Print final stats
        total_time = (datetime.now() - stats['start_time']).total_seconds() / 60
        print(f"\nTranslation completed:")
        print(f"Total contexts processed: {stats['contexts_processed']}/{max_contexts}")
        print(f"Total QA pairs processed: {stats['qa_pairs_processed']}")
        print(f"Total time: {total_time:.1f} minutes")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        # Save progress on error
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    CONFIG_FILE = "D:/Desktop/FY/FYP/QA_Translator/config.json"
    translate_squad_batch(CONFIG_FILE)
