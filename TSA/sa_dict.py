import json
import re
import numpy as np
from dataset import MysqlConnection
from kiwipiepy import Kiwi
import math
import os
import subprocess
from jamo import h2j, j2hcj
import itertools
from tqdm import tqdm
import itertools
from itertools import combinations
from collections import defaultdict
from datetime import datetime, timedelta
from sentiment_dict import NEG_DICT, POS_DICT

# google colab에서 실행할 경우 아래 함수 실행
def install_mecab():
    try:
        current_directory = os.getcwd()
        desired_directory = 'TSA'

        if not current_directory.endswith(desired_directory):
            print(f"Current directory {current_directory} does not match the desired directory {desired_directory}")
            os.chdir(desired_directory)

        subprocess.run(['git', 'clone', 'https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git'])
        os.chdir('Mecab-ko-for-Google-Colab')
        subprocess.run(['bash', 'install_mecab-ko_on_colab_light_220429.sh'])

    except subprocess.CalledProssError as e:
        print(f"Error executing the script: {e}")
    except Exception as e:
        print(f"An error occurred {e}")

    else :
        from konlpy.tag import Mecab  # KoNLPy를 통해 Mecab 패키지 import

def get_jongsung_TF(sample_text):
    sample_text_list = list(sample_text)
    last_word = sample_text_list[-1]
    last_word_jamo_list = list(j2hcj(h2j(last_word)))
    last_jamo = last_word_jamo_list[-1]

    jongsung_tf = "T"

    if last_jamo in ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅘ', 'ㅚ', 'ㅙ', 'ㅝ', 'ㅞ', 'ㅢ', 'ㅐ,ㅔ', 'ㅟ', 'ㅖ', 'ㅒ']:
        jongsung_tf = "F"

    return jongsung_tf


def get_entity_group_word():
    db_connection = MysqlConnection()
    conn = db_connection.connection
    try:
        with conn.cursor() as cursor:
            query = "SELECT word FROM entity GROUP BY word;"
            cursor.execute(query)
            entities = [(entity[0], 0) for entity in list(cursor.fetchall())]
    except pymysql.Error as err:
        print(f"Error: {err}")
    else:
        print(entities)
    finally:
        cursor.close()
        conn.close()


def make_user_dic_csv(morpheme_type, word_list, user_dic_file_name):
  file_data = []

  for word, score in word_list:
    jongsung_TF = get_jongsung_TF(word)

    line = f"{word},,,{score},{morpheme_type},*,{jongsung_TF},{word},*,*,*,*,*\n"

    file_data.append(line)

  with open("./user-dic/user-nnp.csv", 'w', encoding='utf-8') as f:
    for line in file_data:
      f.write(line)


# 불용어 목록 불러오기
def get_stopwords():
    with open("Data/stopwords.txt", 'r') as f:
            stopwords = f.readlines()
            stopwords = [word.strip('\n') for word in stopwords]

    return stopwords

# 새로 정의한 사전 반영
def apply_userdic():
    commands = [
        "bash autogen.sh",
        "sudo make install",
        "bash tools/add-userdic.sh",
        "make clean",
        "make install"
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(e)
            return


# co-occurrence를 활용한 문장 감성분석
# db에서 문장 데이터 불러오기
def load_data(date: str):
  """
  input으로 받은 날짜까지 개체들이 등장했던 문장을 로드함
  """

  db_connection = MysqlConnection()
  conn = db_connection.connection

  try:
    with conn.cursor() as cursor:

      query = "SELECT sentence FROM sentence WHERE date(datetime) <= %s"
      cursor.execute(query, (date, ))  # 적절한 값으로 대체

      # 쿼리 실행 결과 받아오기 -> kw는 다 합쳐서 문자열 처리 해줄 것.
      sents = [sentence[0] for sentence in list(cursor.fetchall())]
      sents = list(set(sents))
      sent_morphs = [mecab.morphs(sents) for sents in tqdm(sents)]

    return sent_morphs

  except pymysql.Error as err:
    print(f"Error: {err}")

  finally:
      # 연결 종료
      cursor.close()
      conn.close()


def generate_co_occurrence_matrix(sent_morphs: list):
    corpus = list(itertools.chain.from_iterable(sent_morphs))
    vocab = set(corpus)
    vocab = list(vocab)
    V = len(vocab)
    N = len(sent_morphs)

    vocab_index = {word: i for i, word in enumerate(vocab)}
    index_vocab = {i: word for i, word in enumerate(vocab)}

    # 문장 내에서 각 단어의 빈도 계산
    word_freq = defaultdict(int)
    for voca in vocab:
      for sent in sent_morphs:
        if voca in sent:
          word_freq[voca] += 1

    freq_matrix = np.zeros(V)

    for i, word in enumerate(vocab):
      freq_matrix[i] = word_freq[word] / N

    product_freq = np.outer(freq_matrix, freq_matrix.T)

    co_occurrence_matrix = np.zeros((V, V))

    for sent in sent_morphs:
      word_indices = [vocab_index[word] for word in sent if word in vocab]
      for a, b in combinations(word_indices, 2):

        if index_vocab[a] not in stopwords and index_vocab[b] not in stopwords: # 불용어는 PMI 행렬에 포함하지 않음

          co_occurrence_matrix[a][b] += 1
          co_occurrence_matrix[b][a] += 1


    co_occurrence_matrix = (co_occurrence_matrix + 1) / (N + 1)


    PMI_matrix = np.log(co_occurrence_matrix / product_freq)
    PMI_matrix = np.maximum(PMI_matrix, 0)
    print(PMI_matrix.shape)

    return PMI_matrix, vocab_index


def analyze_senti(date: str):
  """
 param  : date(string)
  """
  db_connection = MysqlConnection()
  conn = db_connection.connection


  sent_morphs = load_data(date)
  co_occur_matrix, vocab_index = generate_co_occurrence_matrix(sent_morphs)


  try:
    with conn.cursor() as cursor:

      query = """
      SELECT nc_id FROM news_cluster WHERE date(datetime) = %s;
      """
      cursor.execute(query, (date, ))
      ncid_list = [ncid_tuple[0] for ncid_tuple in list(cursor.fetchall())]

      for nc_id in tqdm(ncid_list):

        query = """
        SELECT nid FROM news WHERE nc_id = %s;
        """

        cursor.execute(query, (nc_id, ))
        nid_list = [nid_tuple[0] for nid_tuple in list(cursor.fetchall())]

        for nid in tqdm(nid_list):

          entry_values = []

          query = """
          SELECT e.word, s.eid
          FROM sentence AS s
          JOIN entity AS e ON s.eid = e.eid AND s.nid = e.nid
          WHERE s.nid = %s
          GROUP BY e.main_word, s.eid;
          """

          values = (nid)
          cursor.execute(query, values)

          results = list(cursor.fetchall())

          if results:
            for word, eid in results:
              neg_score = 0
              pos_score = 0
              i = vocab_index[word]

              sorted_indices_desc = np.argsort(-co_occur_matrix[i])

              for key, value in vocab_index.items():
                if key in NEG_DICT:
                  neg_score += co_occur_matrix[i][value]

                elif key in POS_DICT:
                  pos_score += co_occur_matrix[i][value]

              polarity = pos_score - neg_score
              polarity = round(polarity, 4) # 소수점 다섯째자리에서 반올림
              print(f"{word}의 감성 점수: {polarity}")

              entry_values.append((nid, eid, date, polarity))

          query = """
          INSERT INTO sentiment (nid, eid, datetime, polarity) VALUES (%s, %s, %s, %s);
          """
          cursor.executemany(query, entry_values)

          conn.commit()

  except KeyError as err:
    print(f"Key Error : {err}")

  except pymysql.Error as err:
      print(f"DB Error : {err}")

  finally:
      # 연결 종료
     cursor.close()
     conn.close()

# 시작일과 종료일 사이의 날짜 추출
def date_range(start, end):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end-start).days+1)]
    return dates

# 기간 동안의 sentiment 값 추출
def get_sent_of_duration(start, end):
    dates = date_range(start, end)
    result = [analyze_senti(date) for date in dates]
    return result


## test
if __name__=="__main__" :
    install_mecab()
    entities = get_entity_group_word()
    make_user_dic_csv(morpheme_type="NNP", word_list=entities, user_dic_file_name='user-nnp.csv')
    apply_userdic()
    result = get_sent_of_duration('2023-08-17', '2023-08-22')
    print(result)