
# Using Large Language Models to Assess Public Perceptions Around Atrial Fibrillation on Social Media

This repository contains the code for our work using large language models (LLMs) to understand public perceptions around Atrial Fibrillation (AF) from Reddit. 

### Development setup
```sh
conda create -n afib_reddit python=3.11
conda activate afib_reddit
conda install pip
pip install -f requirements.txt
```

### Scraping Reddit data

```sh
python topic_modeling/scrape_reddit.py <afib_db_path>
```

### Topic modeling

```sh
python topic_modeling/topic_modeling.py <afib_db_path> <output_topic_model_file> 
```

### Sentiment analysis

```sh
python topic_modeling/sentiment_analysis <afib_db_path> <output_sentiment_analysis_path>
```
