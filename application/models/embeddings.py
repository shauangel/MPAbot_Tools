import json
import tensorflow_hub as hub
import tensorflow as tf

# url = "https://kaggle.com/models/google/wiki-words/frameworks/TensorFlow2/variations/250/versions/1"
# url = "https://www.kaggle.com/models/google/wiki-words/TensorFlow2/250/1"
url = "/Users/shauangel/.cache/kagglehub/models/google/wiki-words/tensorFlow2/250/1"

def load_embedding_model():
    return


def embeds(token_list):
    embed = hub.KerasLayer(url,
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True,
                           name="Word_Embedding_Layer")
    embeddings = embed(token_list)
    return embeddings
