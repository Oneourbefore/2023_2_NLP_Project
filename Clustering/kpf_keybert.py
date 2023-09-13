# -*- coding: utf-8 -*-
"""
KPF-SBERT로 임베딩하여 코사인 유사도, mmr을 활용한 키워드 모델
기존 KPF-KeyBERT와 달리 명사만 뽑지 않음.
명사만 뽑을 경우 '불합리'가 '불'과 '합리'로 의미가 분리되어 이후 사건 연결 단계에서 방해가 되기 때문임.
"""

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import itertools
import random
import hanja

import mysql.connector
import pymysql

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

custom_stopwords_path = "../Data/stopwords.txt"
n_gram_range = (1, 1)

with open(custom_stopwords_path, "r", encoding="utf-8") as f:
    custom_stopwords = f.read().splitlines()

# CountVectorizer 사용
def make_embeddings(doc: str):

  count = CountVectorizer(ngram_range=n_gram_range).fit([doc])
  candidates = count.get_feature_names_out()

  candidates = [candidate for candidate in candidates if all(candidate not in custom_stopword for custom_stopword in custom_stopwords)]

  # SBERT 모델 로드 및 임베딩
  model = SentenceTransformer('bongsoo/kpf-sbert-v1.1')
  doc_embedding = model.encode([doc])
  candidate_embeddings = model.encode(candidates)

  return doc_embedding, candidate_embeddings, candidates

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def extract_kws(titles):
  """
  한 클러스터로 묶인 기사들의 제목들을 가져와서 키워드 추출
  """

  titles = hanja.translate(titles, 'substitution')

  doc_embedding, candidate_embeddings, candidates = make_embeddings(titles)
  kws = mmr(doc_embedding, candidate_embeddings, candidates, 20, 0.4)
  kws = [kw for kw in kws if not kw.isdigit()][:10] # 숫자로만 이루어진 키워드 제외

  return kws

### mysql 연결 example
from dataset import MysqlConnection # import 구문 있는 곳(상단)에 넣으면 됨
db_connection = MysqlConnection()
conn = db_connection.connection
cursor = conn.cursor()
cursor.execute("")
result = cursor.fetchall()

cursor.close()
conn.close()