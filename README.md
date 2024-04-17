
# Using Large Language Models to Assess Public Perceptions Around Glucagon-Like Peptide-1 Receptor Agonists on Social Media

This repository contains the code for our work using large language models (LLMs) to understand public perceptions around Glucagon-Like Peptide-1 Receptor Agonists (GLP-1-RAs) from Reddit. 

### Development setup
```sh
conda create -n glp1_reddit python=3.11
conda activate glp1_reddit
conda install pip
pip install -f requirements.txt
```

### Scraping Reddit data

```sh
python topic_modeling/scrape_reddit.py <glp1_db_path>
```

### Topic modeling

```sh
python topic_modeling/topic_modeling.py <glp1_db_path> <output_topic_model_file> 
```

### Sentiment analysis

```sh
python topic_modeling/sentiment_analysis <glp1_db_path> <output_sentiment_analysis_path>
```