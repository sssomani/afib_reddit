import pandas as pd
import numpy as np

import re
import argparse
import pickle
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from langchain.llms import Ollama

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler as mms
from sklearn.cluster import KMeans, SpectralClustering
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score

from umap import UMAP

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import matplotlib.font_manager as fm

sns.set_style('whitegrid')
sns.set_context('talk')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = "CMU Sans Serif"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['mathtext.fontset'] = 'cm' 

config = {

    'data' : None,

    'embeddings' : {
        'name' : 'BAAI/bge-base-en-v1.5',
        'device' : 'cuda'
    },

    'save_dir' : None,

    'random_state' : 42,
    
    'topic_hdbscan_params' : {
        'min_cluster_size' : 100,
        'metric' : 'euclidean',
        'cluster_selection_method' : 'eom',
        'prediction_data' : True
    },
    
    'topic_umap_params' : {
        'n_neighbors' : 15,
        'n_components' : 10,
        'min_dist' : 0.0,
        'metric' : 'cosine',
        'random_state' : 42
    },

    'group_umap_params' : {
        'n_neighbors' : 2,
        'n_components' : 3,
        'min_dist' : 0.0,
        'metric' : 'hellinger',
        'spread' : 2,
        'random_state' : 42
    },

    'ngram_representation' : {
        'stop_words' : 'english'
    },


}

class TopicModeling():
    """
    Class for handling BERTopic modeling as it pertains to Reddit. 
    """

    def __init__(self, config=config):
        self.config = config

    def run(self):
        self.load_data_frame()
        self.find_topics()
        self.label_topics()
        self.find_groups()
        self.label_groups()
        self.plot_topics()
        self.create_topic_table()

    def load_data_frame(self):
        """
        Load our Reddit post/comments dataframe.
        """
        if self.config['data'].split('.')[-1] != 'pickle':
            raise TypeError('Expected a pickle file file. Other input dataframe types not yet supported.')

        df = pd.read_pickle(self.config['data'])
        self.df = self.preprocess_dataframe(df)
        self.texts = self.df['content'].to_list()

    @staticmethod
    def preprocess_dataframe(df):
        """
        Preprocess dataframe for topic modeling, if this has not already been performed.
        """
        # Fill empty cells and remove some weird html tags
        df['body'].fillna("", inplace=True)
        df['body'] = df['body'].str.replace("http\S+", "")
        df['body'] = df['body'].str.replace("\\n", " ")
        df['body'] = df['body'].str.replace("&gt;", "")
        
        # Get rid of extra spaces
        df['body'] = df['body'].str.replace('\s+', ' ', regex=True)
        
        # Remove those too small.
        df['body_len'] = df['body'].str.len()
        df = df.query('body_len >= 25')
        return df

    def find_topics(self):
        # Load our vectorizer
        vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Create our embedding model and load the embeddings.
        embedding_model = SentenceTransformer(**self.config['embeddings'])
        self.embeddings = embedding_model.encode(self.texts, show_progress_bar=True, batch_size=64)
        
        # Create our remaining other functions of importance.
        umap_model = UMAP(**self.config['topic_umap_params'])
        u = umap_model.fit_transform(self.embeddings)

        # Create an initial guess of the number of clusters we will likely have and where they are located in the dim-red latent space
        hdbscan_model = HDBSCAN(**self.config['topic_hdbscan_params'])
        clusters = np.array(hdbscan_model.fit_predict(u))

        # Find centroids of all the clusters.
        n_clusters = np.max(clusters) + 1
        centroids = np.empty((n_clusters, u.shape[1]))

        for cluster_i in range(n_clusters):
            inds_in_cluster_i = np.where(clusters == cluster_i)[0]
            points_in_cluster_i = u[inds_in_cluster_i]
            centroids[cluster_i, :] = np.mean(points_in_cluster_i, axis=0)

        # Create our final clustering model
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, init=centroids)
        
        # KeyBERT
        representation_model = KeyBERTInspired()

        # Create our topic model with all of these pieces
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model,
                            embedding_model=embedding_model,
                            umap_model=umap_model,
                            hdbscan_model=kmeans_model,
                            representation_model=representation_model,
                            verbose=True)
        
        self.topics, _ = self.topic_model.fit_transform(self.texts, self.embeddings)

        # Save outputs.
        self.topic_model.save(f'{self.config["save_dir"]}/topic_model.pickle', save_ctfidf=True)
        with open(f'{self.config["save_dir"]}/topics.pickle', 'wb') as fh:
            pickle.dump(self.topics, fh)

    def label_topics(self):
        self.topic_labeler = TopicLabeling(self.df, self.topics, self.embeddings, self.topic_model, self.config)

    def find_groups(self):
        # Normalize the ctfidf representation of topics.
        c_tf_idf_mms = mms().fit_transform(self.topic_model.c_tf_idf.toarray())
        
        # This helps us to visualize
        self.c_tf_idf_vis = UMAP(n_neighbors=2, n_components=2, metric='hellinger', random_state=self.config['random_state']).fit_transform(c_tf_idf_mms)
        self.c_tf_idf_embed = UMAP(**self.config['group_umap_params'], random_state=self.config['random_state']).fit_transform(c_tf_idf_mms)
        
        # Find the ideal # of groups.
        ideal_n_clusters = self.find_ideal_num_groups(self.c_tf_idf_embed)
        self.groups = SpectralClustering(n_clusters=ideal_n_clusters, random_state=self.config['random_state']).fit_predict(self.c_tf_idf_embed) + 1

    def label_groups(self):
        # Label groups
        self.group_labeler = GroupLabeling(self.topics, self.topic_labeler.topic_labels, self.groups)

    def find_ideal_num_groups(self, llim=3, ulim=40):
        
        c_tf_idf_embed = self.c_tf_idf_embed

        ss = []

        cluster_arr = np.arange(llim, ulim, 2)
        
        for n_clusters in cluster_arr:
            clusters = SpectralClustering(n_clusters=n_clusters, random_state=42, n_components=2).fit_predict(c_tf_idf_embed)
            ss.append(silhouette_score(c_tf_idf_embed, clusters))
            # db.append(davies_bouldin_score(c_tf_idf_embed, clusters))
            
        ideal_n_clusters = cluster_arr[np.argmax(ss)]

        print("top silhouette score: {0:0.3f} for at n_clusters {1}".format(np.max(ss), cluster_arr[np.argmax(ss)]))    
        self.ss = ss
        return ideal_n_clusters

    @staticmethod
    def find_clustering_scores(c_tf_idf_embed, llim=3, ulim=40):
    
        ss = []
        db = []
        
        cluster_arr = np.arange(llim, ulim, 2)
        
        for n_clusters in cluster_arr:
            clusters = SpectralClustering(n_clusters=n_clusters, random_state=42, n_components=2).fit_predict(c_tf_idf_embed)
            ss.append(silhouette_score(c_tf_idf_embed, clusters))
            # db.append(davies_bouldin_score(c_tf_idf_embed, clusters))
            
        with sns.plotting_context('notebook'):
            sns.set_style('ticks')
            fig, ax = plt.subplots(figsize=(5, 2.5))
            
            try:
                sns.lineplot(x=cluster_arr, y=ss, palette='autumn', ax=ax, color=cm['autumn'](0.3))
            except Exception as e:
                print(e)
            
            ax.set_ylabel('Silhouette Score')
            ax.set_xlabel('Number of Clusters')
            ax.set_title('Clustering Performance', fontsize=15, y=0.95)

        ideal_n_clusters = cluster_arr[np.argmax(ss)]

        print("top silhouette score: {0:0.3f} for at n_clusters {1}".format(np.max(ss), cluster_arr[np.argmax(ss)]))    
        return ideal_n_clusters

    def plot_topics(self):
        with sns.plotting_context('notebook'):
            sns.set_style('white')
            plt.figure(figsize=(10, 5))

            vis_arr = self.c_tf_idf_vis
            n_clusters = self.groups.max() - self.groups.min() + 1

            ax = sns.scatterplot(x=vis_arr[:, 0], y=vis_arr[:, 1], size=topic_model.get_topic_info()['Count'], \
                            hue=self.groups, \
                            sizes=(100, 5000), \
                            alpha=0.5, palette='tab20', legend=True, edgecolor='k')

            h,l = ax.get_legend_handles_labels()
            legend = plt.legend(h[0:n_clusters],l[0:n_clusters], bbox_to_anchor=(-0.011, -1.95) , loc='lower left', borderaxespad=1, fontsize=10, labels=self.group_labeler.group_labels.values())
            legend.legend_handles[0]._sizes = legend.legend_handles[1]._sizes
            ax.set_title('Topics, Grouped by Similarity of Content', fontsize=16, pad=10)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.figure.savefig(f'{config["save_dir"]}/figure_groups.png', dpi=300, bbox_inches="tight")

        # with sns.plotting_context('notebook'):
        #     sns.set_style('white')
        #     plt.figure(figsize=(10, 9))

        #     vis_arr = points_2d
        #     hue_arr = groups_per_disc

        #     n_clusters = np.max(hue_arr) - np.min(hue_arr) + 1

        #     ax = sns.scatterplot(x=vis_arr[:, 0], y=vis_arr[:, 1], size=0.01, \
        #                     hue=hue_arr, \
        #                     alpha=0.5, palette='Set1', legend=True)

        #     h,l = ax.get_legend_handles_labels()
        #     legend = plt.legend(bbox_to_anchor=(1, -0.02) , loc='lower left', borderaxespad=1, fontsize=10, labels=group_labels_ctfidf_llm.values())
            
        #     ax.set_title('Groups within GLP1RA-Related Discussions', fontsize=16, pad=10)
        #     ax.set_xlim([-10, 10])
        #     ax.set_ylim([-10, 10])
        #     ax.set_xlabel('Feature 1')
        #     ax.set_ylabel('Feature 2')
        #     ax.set_xticklabels([])
        #     ax.set_yticklabels([])

            ax.figure.savefig(f'{config["save_dir"]}/figure_topics_bydisc.png', dpi=300, bbox_inches="tight")

    def create_topic_table(self):
        df, topics, embeddings, topic_labels, groups = self.df, self.topics, self.embeddings, self.topic_labels, self.groups

        rep_docs = self.find_representative_docs_per_topic(df, topics, embeddings, 1)
        topic_table = pd.DataFrame(index=np.arange(1, np.max(topics) + 2).astype('int'), columns=['Discussions (#)', 'Group', 'Topic Label', 'Representative Post'])

        for group in range(1, groups.max() + 1):

            topics_group_i = np.where(groups == group)[0]

            for tl_i in topics_group_i:
                topic_table.loc[tl_i + 1, 'Topic Label'] = topic_labels[tl_i]
                topic_table.loc[tl_i + 1, 'Group'] = group
                topic_table.loc[tl_i + 1, 'Representative Post'] = rep_docs[tl_i][0]
                topic_table.loc[tl_i + 1, 'Discussions (#)'] = len(np.where(np.array(topics) == tl_i)[0])

        topic_table.to_excel(f'{config["save_dir"]}/topic_table.xlsx')

    def load_topic_model(self, path_to_topic_model):
        # Load a saved topic model.
        raise NotImplementedError


class TopicLabeling():

    def __init__(self, df, topics, embeddings, topic_model, config):
        self.df = df
        self.topics = topics
        self.embeddings = embeddings
        self.topic_model = topic_model
        self.config = config

        self.llm = Ollama(model="llama2")

        self.topic_representations = self.find_topic_representations(topic_model, df, topics, embeddings)
    
    def find_topic_representations(self, n_reps=10):
        df = self.df
        topics = self.topics
        embeddings = self.embeddings
        topic_model = self.topic_model

        # Find representative documents for each topic
        rep_docs = self.find_representative_docs_per_topic(df, topics, embeddings, n_reps)

        # Find a random selection of documents for each topic
        rand_docs = self.find_random_docs_per_topic(df, topics, n_reps)
        
        # Combine these two
        prompt_docs = [i[:5] + j[:5] for i, j in zip(rep_docs, rand_docs)]

        # Find the representation for each topic
        representations = {}
        
        for topic in tqdm(range(np.max(topics) + 1)):
            prompt_i = self.prepare_prompt(topic_model, prompt_docs, topic)
            representations[topic] = self.llm.predict(prompt_i)

        self.topic_representations = representations
        with open(f'{config["save_dir"]}/llama_topic_representations_i.pickle', 'wb') as fh:
            pickle.dump(representations, fh)

        pattern = r'(?<=\d(.|:)\s)(.*?)(?=(\\n)+(?:Label )?\d+(.|:)|(\n|\Z))'
        self.representations_extracted = {topic : [i[1].replace('"', '') for i in re.findall(pattern, representation)] for (topic, representation) in representations.items()}
        self.topic_labels = ['. '.join(representations) for _, representations in self.representations_extracted.items()]

    def prepare_topic_results_for_review(self):
        df = self.df
        topics = self.topics
        embeddings = self.embeddings
        topic_model = self.topic_model
        topic_representations = self.topic_representations
        
        rep_docs = self.find_representative_docs_per_topic(df, topics, embeddings, 5)
        rand_docs = self.find_random_docs_per_topic(df, topics, 5)

        final_prompt = ''

        for topic in tqdm(range(np.max(topics) + 1)):
            
            prompt = f'\n================================ TOPIC {topic} ======================================\n'
            
            prompt += 'Topic keywords:\n\''
            prompt += '\', \''.join(topic_model.get_topic_info(topic)['Representation'][0])
            
            prompt += '\'.\n\nRepresentative discussions:\n=======' 
            prompt += '\n======='.join(rep_docs[topic])

            prompt += '\'.\n\nRandom discussions:\n=======' 
            prompt += '\n======='.join(rand_docs[topic])

            topic_labels = ', '.join(topic_representations[topic])
            prompt += f'\n\nFinal label: {topic_labels}'

            final_prompt += prompt + '\n===========================================================================================\n'

        return final_prompt
    
    @staticmethod
    def prepare_prompt(topic_model, rep_docs, topic_of_interest):
        # Create prompt for LLaMa2
        prompt = 'You are a honest, scientific chatbot that helps me, a Cardiologist, create unique, diverse labels for a topic based on representative discussions and keywords. Do not be creative or loquacious. Please present the topic label in a short and direct manner.\n\n'
        prompt += 'I have a topic that is described by the following keywords:\n\''
        prompt += '\', \''.join(topic_model.get_topic_info(topic_of_interest)['Representation'][0])
        prompt += '\' .\n\nIn this topic, the following documents are a small but representative subset of all other documents in the topic:\n\n' 
        prompt += '\n\n'.join(rep_docs[topic_of_interest])
        prompt += '\n\nBased on the information above, can you create three short, direct labels without descriptions for this topic?'

        return prompt

    @staticmethod
    def find_representative_docs_per_topic(df, topics, embeddings, n_reps=5):
        topics = np.array(topics)
        n_topics = topics.max() + 1
        emb_dim = embeddings.shape[1]
        samples_in_topics = [np.where(topics == i)[0] for i in range(n_topics)]
        centroids = np.array([np.mean(embeddings[topic_inds, :], axis=0) for topic_inds in samples_in_topics])
        
        representative_samples = []
        
        for topic_i, (centroid_i, samples_i) in tqdm(enumerate(zip(centroids, samples_in_topics))):
            embedded_samples_i = embeddings[samples_i, :]
            distances = cosine_distances(embedded_samples_i, centroid_i.reshape(1, emb_dim)).flatten()
            dist_inds = np.argsort(distances)
            rep_docs_i = df.iloc[samples_i[dist_inds[:n_reps]]]['body'].values.tolist()
            representative_samples.append(rep_docs_i)
        
        return representative_samples

    @staticmethod
    def find_random_docs_per_topic(df, topics, n_reps):
        rand_docs = []
        for topic in tqdm(range(np.max(topics) + 1)):
            topic_inds = np.where(np.array(topics) == topic)[0]
            if len(topic_inds) < n_reps:
                samples_i = df.iloc[topic_inds]['body'].values.tolist()
            else:
                samples_i = df.iloc[topic_inds]['body'].sample(n=n_reps, random_state=42).values.tolist()
            rand_docs.append(samples_i)
        return rand_docs
    
    def prepare_prompts_for_topic_labeling(self, n_reps=10):

        df = self.df
        topics = self.topics
        embeddings = self.embeddings
        topic_model = self.topic_model


        # Find representative documents for each topic
        rep_docs = self.find_representative_docs_per_topic(df, topics, embeddings, n_reps)

        # Find a random selection of documents for each topic
        rand_docs = self.find_random_docs_per_topic(df, topics, n_reps)
        
        # Combine these two
        prompt_docs = [i[:5] + j[:5] for i, j in zip(rep_docs, rand_docs)]

        # Find the representation for each topic
        prompts = {}
        for topic in tqdm(range(np.max(topics) + 1)):
            prompt_i = self.prepare_prompt(topic_model, prompt_docs, topic)
            prompts[topic] = prompt_i

        self.prompts = prompts

class GroupLabeling():

    def __init__(self, topics, topic_labels, groups):
        self.topics = topics

        self.create_group_labels(groups, topic_labels)
        self.prepare_prompts_for_group_labeling()
        self.finalize_group_labels_with_llm()

    def create_group_labels(self, groups, topic_labels):
        group_labels_combined = {}

        g_min = np.min(groups)
        g_max = np.max(groups)
        for group_i in range(g_min, g_max + 1):
            group_i_inds = np.where(group_i == groups)[0]
            group_i_label = ''
            for group_i_ind in group_i_inds:
                group_i_label += topic_labels[group_i_ind].replace('"', '') + '\n'
            group_labels_combined[group_i] = group_i_label

        for group_i, group_label_combined in group_labels_combined.items():
            print(f'{group_i} :: {group_label_combined}')

        self.group_labels_combined = group_labels_combined

    def prepare_prompts_for_group_labeling(self):
        group_prompts = {}
        for group, group_label in self.group_labels_combined.items():
            # Create prompt for LLaMa2
            prompt = 'You are a honest, scientific chatbot that helps me, a Cardiologist, create a single representative label for a group that represents a series of topics based on topic labels. Each topic label is separated by a new line character. Do not be creative or loquacious. Please present the group label in a short and direct manner.\n\n'
            prompt += 'I have a group that is described by the following topic labels:\n\''
            prompt += group_label
            prompt += '\n\nBased on the information above, can you create the one, best, direct label for this topic in the following format?\n'
            prompt += 'Group Label: <group_label>'

            group_prompts[group] = prompt
            
        self.group_prompts = group_prompts
    
    def finalize_group_labels_with_llm(self):
        labels = {}
        llm = Ollama(model="llama2", temperature=0.1)
        pattern = r"(?<=Group Label: )(.*)"
        
        for group, prompt in tqdm(self.group_prompts.items()):
            response = llm.predict(prompt)
            response.replace('"', '')
            labels[group] = re.findall(pattern, response)[0]
            
        self.group_labels = labels
    
    def create_topic_group_listing(self):

        groups, topic_labels = self.groups, self.topic_labels

        text = ''

        for group in range(1, groups.max() + 1):
            text += f'=========================== GROUP {group} ===========================\n'
            text += f'LLM Label: {self.group_labels[group]}\n'
            text += f'==== TOPICS ====\n'
            topics_group_i = np.where(groups == group)[0]

            for tl_i in topics_group_i:
                text += f'TOPIC {tl_i} :: {topic_labels[tl_i]}\n'

            text += f'===============================================================\n'

        return text

if __name__ == '__main__':
    # Enter input data via argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type='str', default='data/afib_db.xlsx', help='Path to dataset')
    parser.add_argument('output', type='str', default='data/topic_model_res.xlsx', help='Path to save topic, group labels')
    args = parser.parse_args()

    config['data'] = args.data
    config['output'] = args.output

    topic_model = TopicModeling(config)
    topic_model.load_data_frame()
    topic_model.create_topic_model()

