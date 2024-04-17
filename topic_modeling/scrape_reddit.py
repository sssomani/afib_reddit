import pandas as pd
import json
import requests
import os
from urllib.parse import quote as url_parse
import argparse


def create_base_url(discussion, search_string):
    """
    Function to create the base URL for scraping Reddit.
    """
    
    # Ensure specified discussion is either a post (submission) or comment.
    assert discussion in ['submission', 'comment']
    
    # Craft our URL.
    return f'https://api.pullpush.io/reddit/search/{discussion}/?q={search_string}&size=100'

def scrape_to_json(url):
    """
    Function that scrapes the Pushshift API based on a specific URL and converts those results to a Pandas dataframe.
    """
    
    data = requests.get(url)
    data_json = json.loads(data.content)
    
    try:
        data_pd = pd.DataFrame(data_json['data'])
    except:
        return None
    
    return data_pd

def scrape_reddit(base_url, save_dir, search_string, type=None):
    
    keys_to_keep = [
        'subreddit',
        'author',
        'created_utc',
        'body',
        'id'
    ]
    
    if os.path.exists(f'{save_dir}db_{search_string}_{type}.pickle'):
        discussions = pd.read_pickle(f'{save_dir}db_{search_string}_{type}.pickle')
        print(f'Old discussions file loaded! Number of discussions: {len(discussions)}')
    else:
        print('Discussions file not found; restarting.')
        discussions = scrape_to_json(base_url)

    n_discussions = len(discussions)
    
    while n_discussions > 0:
        last_utc = int(discussions['created_utc'].iloc[-1])
        url = base_url + f'&after=0&before={last_utc}'
        
        next_discs = scrape_to_json(url)
        if next_discs is None:
            print('No discussions found, so let\'s try this again...')
            continue

        n_discussions = len(next_discs)
                
        discussions = pd.concat([discussions, next_discs])
    
        if 'submission' in base_url:
            discussions['body'] = discussions['title'] + '. ' + discussions['selftext']

        print(f'Total {type} so far: {discussions.shape[0]}')
        discussions.to_pickle(f'{save_dir}db_{search_string}_{type}.pickle')
            
    
    return discussions[keys_to_keep]

def get_reddit_data(search_strings, save_dir):
    '''
    
    Scrape Reddit for all discussions related to a set of search strings.         

    Parameters
    ----------
    search_string : str
        Search string to query.
    output_fn : str, optional
        Name of the local Excel database to save data. 

    Returns
    -------
    posts_df : Pandas dataframe
        Database

    '''
    
    discussions = pd.DataFrame(columns=[
        'subreddit',
        'author',
        'created_utc',
        'body',
        'type'
        'id',
    ])
    
    for search_string in search_strings:
        print(f'========= {search_string} ==========')

        search_string = url_parse(search_string)

        # First, let's start by searching all posts.
        post_url = create_base_url('submission', search_string)
        posts = scrape_reddit(post_url, save_dir, search_string, type='posts')
        posts['type'] = 'post'
        posts['search_string'] = search_string

        # Now, let's search all comments.
        comment_url = create_base_url('comment', search_string)
        comments = scrape_reddit(comment_url, save_dir, search_string, type='comments')
        comments['type'] = 'comment'
        comments['search_string'] = search_string
        
        discussions = pd.concat([discussions, posts, comments])
    
    discussions.drop_duplicates(subset='body', inplace=True)
    print(f'Total of {discussions.shape[0]} found!')

    return discussions

def preprocess_db(df):
    # Fill empty cells and remove some weird html tags
    df['body'].fillna("", inplace=True)
    df['body'] = df['body'].str.replace("http\S+", "")
    df['body'] = df['body'].str.replace("\\n", " ")
    df['body'] = df['body'].str.replace("&gt;", "")
    df['body'] = df['body'].str.replace('\s+', ' ', regex=True)
    df['body_len'] = df['body'].str.len()
    df = df.query('body_len >= 25')
    df.reset_index(inplace=True)
    df = df.drop('index', axis=1)
    return df

if __name__ == '__main__':

    """
    Scrape Reddit for all GLP1-RA-related keywords.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type='str', help='Path to save dataset')
    args = parser.parse_args()

    glp1_strings = ['semaglutide', 'rybelsus', 'wegovy', 'ozempic',
                    'retatrutide',
                    'dulaglutide', 'trulicity',
                    'tirzepatide', 'mounjaro',
                    'liraglutide', 'saxenda',
                    'exenatide', 'bydureon', 'byetta',
                    'lixisenatide', 'adlyxin']

    glp1_db = get_reddit_data(glp1_strings, args.output_dir)
    glp1_db = preprocess_db(glp1_db)
    glp1_db.to_pickle(args.output_dir + 'full_db.pickle')

    try:
        glp1_db.to_excel(args.output_dir + 'full_db.xlsx')
    except:
        print('Failed to export Excel file.')