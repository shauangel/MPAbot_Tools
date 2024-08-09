# Created in 2021
# Block Analysis v.0
#
#

import numpy as np
import nltk
import spacy
# LDA model
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from itertools import chain


# TextAnalyze Module: pre-processing word content, ex
class TextAnalyze:
    STOPWORDS = []  # 停用詞: 可忽略的詞，沒有賦予上下文句意義的詞
    POS_TAG = ['PROPN', 'ADJ', 'NOUN', 'VERB']  # 欲留下的詞類noun
    WHITE_LIST = ['pandas']

    def __init__(self):
        try:
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
        except Exception:
            nltk.download('stopwords')
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
        self.STOPWORDS += ['use', 'python']
        return

    # 文本前置處理
    def content_pre_process(self, text):
        nlp = spacy.load('en_core_web_sm')
        # Step 1. lowercase & tokenizing
        doc = nlp(text.lower())
        # Step 2. reduce punctuation
        pure_word = [token for token in doc if not token.is_punct and token.text != '\n']
        # Step 3. pos_tag filter & lemmatize
        lemma = []
        for token in pure_word:
            if token.pos_ in self.POS_TAG:
                if token.lemma_ == "-PRON-":
                    lemma.append(token.text)
                else:
                    lemma.append(token.lemma_)
        lemma = list(dict.fromkeys(lemma))  # reduce duplicate words

        # Step 4. reduce stopwords & punctuation
        filtered_token = [word for word in lemma if not nlp.vocab[word].is_stop or word in self.STOPWORDS]

        return filtered_token, doc

    @staticmethod
    def lda_topic_modeling(data, topic_num):
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(text) for text in data]
        lda_model = LdaModel(corpus, num_topics=topic_num, id2word=dictionary, per_word_topics=True)
        return lda_model, dictionary

    # 關聯度評分
    # input(question keywords, pure word of posts' question)
    def similarity_ranking(self, question_key, compare_list):
        nlp = spacy.load('en_core_web_lg')
        # pre-process text
        comp_preproc_list = [self.content_pre_process(content)[0] for content in compare_list]
        # LDA topic modeling
        lda_model, dictionary = self.lda_topic_modeling(comp_preproc_list, 5)

        # topic prediction
        q_bow = dictionary.doc2bow(question_key)
        q_topics = sorted(lda_model.get_document_topics(q_bow), key=lambda x: x[1], reverse=True)

        # choose top 3 prediction
        top3_topic_pred = [q_topics[i][0] for i in range(3)]  # top3 topic
        top3_prob = [q_topics[i][1] for i in range(3)]  # top3 topic prediction probability
        print(top3_prob)
        top3_topic_keywords = [" ".join([w[0] for w in lda_model.show_topics(formatted=False, num_words=5)[pred_t][1]])
                               for pred_t in top3_topic_pred]
        print(top3_topic_keywords)
        q_vec_list = [nlp(keywords) for keywords in top3_topic_keywords]
        top3pred_sim = [[q_vec.similarity(nlp(" ".join(comp))) for comp in comp_preproc_list] for q_vec in q_vec_list]
        top3pred_sim = np.array(top3pred_sim)
        print(np.array([top3pred_sim[i] * top3_prob[i] for i in range(3)]))
        score_result = np.sum(np.array([top3pred_sim[i] * top3_prob[i] for i in range(3)]), axis=0)
        return score_result


def block_ranking(stack_items, qkey):
    a = TextAnalyze()
    ans = [items['answers'] for items in stack_items]

    # data pre-process
    all_content = [[{"id": sing_ans["id"], "content": sing_ans['content']}
                    for sing_ans in q_ans_list] for q_ans_list in ans]
    all_content_flat = list(chain.from_iterable(all_content))
    raw = [t["content"] for t in all_content_flat]

    # similarity ranking
    temp_result = a.similarity_ranking(qkey, raw)
    for i in range(len(all_content_flat)):
        all_content_flat[i]["score"] = temp_result[i]
    rank = sorted(all_content_flat, key=lambda data: data["score"], reverse=True)
    return rank[:10]


if __name__ == "__main__":

    """
    questions, responses = get_filename()
    analyzer = TextAnalyze()
    for idx in range(len(questions)):
        # set loggers
        logging.basicConfig(level=config.LOG_MODE, force=True,
                            filename="logs/old_method/" + str(idx + 1) + "/result.log", filemode='w',
                            format=config.FORMAT, datefmt=config.DATE_FORMAT)
        # get parse posts
        with open("stack_data/" + responses[idx], "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
            titles = [i['question']['title'] for i in data]
            raw_file.close()

        # pre-process user question
        key = analyzer.content_pre_process(questions[idx])[0]

        # start block ranking process
        r = block_ranking(data, key)
        for detail in r:
            print(detail)
            logging.info(detail)
    """