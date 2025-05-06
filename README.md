# SQuAD v1.0 Sinhala Translation Script

This repository provides scripts to translate the [SQuAD v1.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset into **Sinhala** using the **Google Translate API**, and to **clean and validate** the resulting QA data by correcting answer spans and removing invalid entries.

## Features

- Translates **context**, **question**, and **answer** fields to Sinhala.
- Recalculates answer start indices post-translation.
- Filters out QA pairs where the translated answer is not found in the translated context.
- Preserves the SQuAD format for compatibility with existing QA models and tools.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/j-ranasinghe/SQuAD-Translation.git
cd SQuAD-Translation
```
### 2. 2. Install Required Dependencies
```
pip install -r requirements.txt
```
### 2. Add Google Translate API Credentials
- Go to Google Cloud Console.
- Enable the Cloud Translation API.
- Create a service account.
- Download the key and save it as credentials.json in the project root.
### 4. Configure Paths:

🚀 Running the Translation 
```
python translation_pipeline.py
```
🧹 Cleaning the Translations - To clean the translated dataset (e.g., fix answer spans, remove invalid answers):
```
  python clean_translations.py
```
