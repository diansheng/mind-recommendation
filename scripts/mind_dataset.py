
import pandas as pd
import numpy as np

"""
There are multiple format of the dataset, depending how the model will use it.


#### user_id, item_id, label
Can be used as binary classification problem. Use two-tower model for training.
If use dot or cosine product, casting user and item into the same latent space. Learn entity embedding based on entity id.
If use mutual attention, can be a basic classifier.

#### user_id, item_id_pos, item_id_neg 
Contrastive learning with triplet loss. Learn entity embedding based on entity id.

#### user_id, item_id_pos, item_id_neg1, item_id_neg2, etc... Negative are from non-clicked impressions
Multi-class classification. Learn entity embedding based on entity id.

Add user context and item context brings more variations


#### Add user click history
Make use of history click, DIN. 

#### item_context
Unsupervised learning. Learn item embedding

#### user_id, item_id
Can be used for deepwalk, graph embedding, node2vec

#### user, item_a, item_b
SwingI2I

#### user_a, item_id, user_b
SwingU2U. User clustering.

"""



USER_ID, NEWS_ID, IMPRESSION_ID, TIMESTAMP, HISTORY, IMPRESSIONS, LABEL = 'user_id', 'news_id', 'impression_id', 'timestamp', 'history', 'impressions', 'label'
CATEGORY, SUB_CATEGORY, TITLE, ABSTRACT, URL, TITLE_ENTITY, ABSTRACT_ENTITY = 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'


def load_behaviors_data(file_name):
    df_behaviors = pd.read_csv(file_name,sep='\t',header=None)
    cols = [IMPRESSION_ID, USER_ID, TIMESTAMP ,HISTORY, IMPRESSIONS]
    df_behaviors.columns = cols
    print(f'{file_name}, data shape: {df_behaviors.shape}')
    return df_behaviors


def load_news_data(file_name):
    df_news = pd.read_csv(file_name,sep='\t',header=None)
    cols = [NEWS_ID, CATEGORY, SUB_CATEGORY, TITLE, ABSTRACT, URL, TITLE_ENTITY, ABSTRACT_ENTITY]
    df_news.columns = cols
    print(f'{file_name}, data shape: {df_news.shape}')
    return df_news


def load_data(env):
    behaviors = load_behaviors_data(f'../data/{env}/behaviors.tsv')
    news = load_news_data(f'../data/{env}/news.tsv')
    return behaviors, news


def behavior_to_user_item_pair_w_label(df):
    # explode impressions into separated rows
    df['impressions2'] = df[IMPRESSIONS].str.split(' ')
    df = df.explode('impressions2')
    
    # build labels
    df['label'] = df['impressions2'].apply(lambda x: int(x[-1])).astype(np.uint8)
    df['news_id'] = df['impressions2'].apply(lambda x: x[:-2])
    return df[[USER_ID, NEWS_ID, LABEL]]


def get_label_from_behavior(df):
#     df['is_clicked'] =
    return df['impressions'].apply(lambda s: [int(x[-1]) for x in s.split(' ')])


def predict_from_behavior(df):
    def _calculate_relevence(r):
        news = [x.split('-')[0] for x in r['impressions'].split(' ')]
        global news_2_vec, user_2_vec
        
        # if user id is not calculated, pad 0 as relevence
        if r['user_id'] not in user_2_vec:
            return np.zeros(len(news))
        
        # construct similarity
        user_v = user_2_vec.get(r['user_id'])
        relevence = [cosine_sim(news_2_vec.get(news_id),user_v) if news_id in news_2_vec else 0 for news_id in news ]
        
        rank = np.argsort(np.argsort(relevence)[::-1]) + 1  # really trick i would say. check https://github.com/numpy/numpy/issues/8757
        return 1./rank
    return df.progress_apply(_calculate_relevence, axis=1)
  
    
    
def load_w2v_from_file(glove_embedding_file, dim_size=None, slim=False):
    """
    Load word2vec
    
    glove_embedding_file: file name of the embedding
    dim_size: 50, 100, 200, 300
    return embedding matrix with its corresponding word_to_index dictionary
    """
    df = pd.read_csv(glove_embedding_file, sep=" ", quoting=3, header=None, index_col=0)
#     glove = {key: val.values for key, val in df.T.items()}
    
    vocab = []
    emb = []
    
    # initialize dim size if needed
    if dim_size is None:
        dim_size = df.shape[1]
        
    emb.insert(0, list(np.random.randn(dim_size)))  # insert a randomized embedding for padding
      
    for key, val in df.T.items():
#         assert len(val.values)==dim_size
        vocab.append(key)
        emb.append(val.values)
    
    word_ind = {w:i+1 for i,w in enumerate(vocab)}  # index starts from 1, reserve 0 for padding
    
    # change to user friendly
    vectors = np.array(emb,dtype=np.float32)
#     weight = torch.FloatTensor(emb)
#     embedding, word_ind  = load_w2v_from_file(glove_embedding_file, dim_size=None)
    
    return vectors, word_ind 
