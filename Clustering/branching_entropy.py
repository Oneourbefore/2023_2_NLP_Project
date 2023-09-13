# -*- coding: utf-8 -*-
"""Copy of BranchingEntropy_def.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H9UX9wQ475MhacxyJrmKw2DSOW4khari
"""


# 불용어 로드
with open('../Data/stopwords.txt', 'r') as f:
    stopwords = f.readlines()
    stopwords = [word.strip('\n') for word in stopwords]

from kiwipiepy import Kiwi
import pandas as pd
import pymysql
import urllib.request
from soynlp.word import WordExtractor
import ast
from konlpy.tag import Komoran
from database import MysqlConnection

kiwi = Kiwi()
komoran = Komoran()


def word_score(score):
    return (score.right_branching_entropy)

def noun_extractor(text: str):
    results = []
    result = kiwi.analyze(text)
    for token, pos, _, _ in result[0][0]:
      if token not in stopwords:
        if len(token) != 1:
          if pos.startswith('N') or pos.startswith('SL'):
            results.append(token)

    return results

def load_sum_data(nc_id: str):
  """
  날짜와 테이블을 지정하여 DB에서 데이터를 로드하는 함수.
  테이블 이름이 NewsCluster일 경우 클러스터 id를
  date는 '2023-08-08' 형식으로
  """
  db_connection = MysqlConnection()
  conn = db_connection.connection

  try:
    with conn.cursor() as cursor:
        # 데이터를 받아올 SQL 쿼리 작성
        query = "SELECT summary FROM news WHERE nc_id = %s;"
        cursor.execute(query, (nc_id,))  # 적절한 값으로 대체

        # 쿼리 실행 결과 받아오기
        results = list(cursor.fetchall())

        results = [result[0] for result in results]

        return results

  except pymysql.Error as err:
    print(f"Error: {err}")

  finally:
      # 연결 종료
      cursor.close()
      conn.close()

def make_event_name(nc_id: str):
    summary_list = load_sum_data(nc_id)

    # 문장 단위 리스트로 만들기
    sentence_list = []

    for doc in summary_list:
        sentences = doc.split('.')  # '.' 뒤에 공백을 포함하여 문장 분리
        sentence_list = sentence_list + sentences

    word_extractor = WordExtractor(min_frequency=5,
        min_cohesion_forward=0.05,
        min_right_branching_entropy=0
    )
    word_extractor.train(sentence_list) # noun_sens_list or sentence_list
    words = word_extractor.extract()

    words = {key: value for key, value in words.items() if len(key) > 1}

    data = []
    for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:]:
        # if score.cohesion_forward
        data.append((word, score.leftside_frequency, score.cohesion_forward, score.right_branching_entropy))

    # 빈도수가 같은 단어 중에서 길이가 가장 큰 단어를 저장할 딕셔너리
    unique_words = {}
    for word, freq, cohesion, entropy in data:
        if not any(len(word) < len(other_word) and word in other_word for other_word, freq, cohesion, entropy in data):
            # unique_words.append((word, freq, cohesion, branching))
            unique_words[word] = (freq, cohesion, entropy)


    df = pd.DataFrame.from_dict(data=unique_words, orient='index')

    df = df.reset_index()
    df.columns=['word','freq','cohesion','brancing_entropy']
    len_sort_df = df.sort_values(by='word', key=lambda x: x.str.len(), ascending=False)
    len_sort_df.reset_index(drop=True, inplace=True)

    # 단어들에 대해 품사 태깅해서 josa 붙은 거 제거
    len_sort_df['pos'] = len_sort_df['word'].apply(lambda x: komoran.pos(x))
    filtered_len_df = len_sort_df[len_sort_df['pos'].apply(lambda x: not any(item[1] in ['JKS','JC', 'JX','JKQ', 'JKV','JKC','JKG', 'JKO', 'JKB','EC','MAG'] for item in x))]
    filtered_len_df = filtered_len_df[~filtered_len_df['word'].isin(stopwords)]

    # stopwords 제거
    filtered_len_df = filtered_len_df[filtered_len_df['pos'].apply(lambda x: not any(item[0] in stopwords for item in x))]
    filtered_len_df['word_len'] = filtered_len_df['word'].apply(lambda x: len(x))
    output_df = filtered_len_df.sort_values(by=['freq','brancing_entropy','word_len'] , ascending=[False,True, False])[:3]

    output_dict = {}  # 변경: 빈 딕셔너리 생성
    output_dict[nc_id] = ','.join(output_df['word'])

    return output_dict