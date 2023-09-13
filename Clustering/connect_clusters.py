from google.colab import drive
drive.mount('/content/drive/')

import os
from datetime import datetime, timedelta
import torch
from tqdm import tqdm

import mysql.connector
import pymysql
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from branching_entropy import make_event_name

"""### Load data"""
def load_data(date: str):
  """
  해당 날짜의 데이터를 불러와서 {nc_id: [kws]} dict을 만듦
  """
  pass

def date_range(end_date: str):
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    start = end_date - timedelta(days=4)
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end_date - start).days + 1)]

    # 2023년 7월 24일 이전의 날짜를 제외하고 반환
    filtered_dates = [date for date in dates if date >= "2023-07-24"]

    return filtered_dates

def iso_con_sbert(today: str):
    """
    SBERT로 임베딩하여 키워드끼리 비교.
    코사인 유사도가 임계치를 넘으면 클러스터를 사건에 편입시키는 함수
    임계치를 넘지 못한 오늘의 클러스터는 새로운 사건이 됨.
    오늘 날짜는 2023-07-24 이후여야 함.
    """

    # Load SBERT model
    model = SentenceTransformer('bongsoo/kpf-sbert-v1.1')

    threshold = 0.5

    dates = date_range(today)
    today_results = load_data(today)

    total_ncids = {nc_id: -1 for nc_id in today_results.keys()}

    for today_ncid, today_kw in today_results.items():
      best_score = 0
      day_scores = []

      for past_date in dates[:-1]: # ['2023-07-24', '2023-07-25'...] '2023-07-30'
        past_results = load_data(past_date)

        for past_ncid, past_kw in past_results.items():
          score = cosine_similarity([model.encode(today_kw)], [model.encode(past_kw)])

          if score >= threshold and score >= best_score:
            total_ncids[today_ncid] = past_ncid
            best_score = score

    return total_ncids

def issue_events(today: str):
    """
    오늘의 클러스터와 과거의 클러스터를 잇는 함수.
    iso_con 함수와 date_range 함수를 이용해서 일주일 간의 클러스터와 오늘의 클러스터를 비교.
    today는 '2023-07-24' 형식으로
    """

    new_events = []  # 새로운 사건이 될 후보 클러스터
    total_ncids = iso_con_sbert(today)  # total_ncids는 {1: 6, 2: 6, 3: 7, 4: -1, 5: -1}의 형태

    for today_nc_id, past_nc_id in total_ncids.items():  # connected_today_nc_ids are list
        if past_nc_id != -1:  # 과거의 클러스터와 연결되었음.
            # 연결된 past_nc_id의 cid를 DB에 업데이트
            pass

    nested_list = list()  # 과거의 사건과 연결되지 않은 클러스터들 리스트

    if nested_list:
        flattened_list = [ncid_tuple[0] for ncid_tuple in nested_list]
        new_events.extend(flattened_list)

    for new_event_ncid in new_events:  # 어떤 사건에도 연결되지 않았으면 새로운 사건으로 등록
        # 이름 짓기: 최초 클러스터의 요약 이용
        event_name = make_event_name(new_event_ncid)[new_event_ncid]

"""## 예시 결과 저장하기"""

def visualize_iso_con(today: str):
    """
    키워드끼리 비교.
    bertscore가 임계치를 넘으면 클러스터를 사건에 편입시키는 함수
    임계치를 넘지 못한 오늘의 클러스터는 새로운 사건이 됨.
    오늘 날짜는 2023-07-24 이후여야 함.
    """
    # Load SBERT model
    model = SentenceTransformer('bongsoo/kpf-sbert-v1.1')

    threshold = 0.5

    dates = date_range(today)
    today_results = load_data(today)
    total_result = []

    total_ncids = {nc_id: -1 for nc_id in today_results.keys()}

    for today_ncid, today_kw in today_results.items():
      best_score = 0

      for past_date in dates[:-1]: # ['2023-07-24', '2023-07-25'...] '2023-07-30'
        past_results = load_data(past_date)

        for past_ncid, past_kw in past_results.items():
          score = cosine_similarity([model.encode(today_kw)], [model.encode(past_kw)])
          result = [today_ncid, today_kw, past_ncid, past_kw, score[0][0]]
          total_result.append(result)

          if score >= threshold and score >= best_score:
            total_ncids[today_ncid] = past_ncid
            best_score = score

    return total_result

### mysql 연결 example
from database import MysqlConnection # import 구문 있는 곳(상단)에 넣으면 됨
db_connection = MysqlConnection()
conn = db_connection.connection
cursor = conn.cursor()
cursor.execute("")
result = cursor.fetchall()

cursor.close()
conn.close()