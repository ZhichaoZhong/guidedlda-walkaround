import numpy as np
from guidedlda import guidedlda as glda
from guidedlda import glda_datasets as gldad

X = gldad.load_data(gldad.NYT)
vocab = gldad.load_vocab(gldad.NYT)
word2id = dict((v, idx) for idx, v in enumerate(vocab))
print(X[:10])




print("TESTING....")

seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
                   ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                   ['music', 'write', 'art', 'book', 'world', 'film'],
                   ['political', 'government', 'leader', 'official', 'state', 'country',
                    'american','case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

model = glda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)

seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)





n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))