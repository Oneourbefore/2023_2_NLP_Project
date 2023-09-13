import re
import json
import kss
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from datetime import datetime, timedelta
from scrap_naver_news import 멀티정치기사긁어오기, 멀티사회기사긁어오기, 언론사별사회기사긁어오기, news_to_json

def remove_special_characters(text):
    # 숫자 사이의 소숫점은 유지하고, 모든 점은 삭제
    text = re.sub(r"(?<=\d)\.(?=\d)", '|', text)  # 숫자 사이의 점을 |로 대체
    p_email = re.compile('[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.([A-Z|a-z]{2,})?')
    text = p_email.sub('', text)
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9\s|.]", " ", text)  # 숫자, 알파벳, 공백, |를 제외한 모든 문자 제거
    text = re.sub(r'  ', ' ', text)
    text = text.replace('|', '.')  # |를 다시 원래의 점으로 변경
    return text

# 추가된 불용어 제거해 주었음
def remove_stopwords(text, stopwords):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# SBERT 모델 로드
sbert_model = SentenceTransformer('bongsoo/kpf-sbert-v1.1')

def cal_similarity(title, content):
    title_embedding = sbert_model.encode(title, convert_to_tensor=True)
    sen_list = kss.split_sentences(content)

    top_sentences = []
    top_similarities = []

    for sen in sen_list:
        sen_embedding = sbert_model.encode(sen, convert_to_tensor=True)
        cosine_similarity = torch.nn.functional.cosine_similarity(title_embedding, sen_embedding, dim=0)

        # Update the top 3 sentences and their similarities
        if len(top_sentences) < 3:
            top_sentences.append(sen)
            top_similarities.append(cosine_similarity.item())
        else:
            min_index = top_similarities.index(min(top_similarities))
            if cosine_similarity > top_similarities[min_index]:
                top_sentences[min_index] = sen
                top_similarities[min_index] = cosine_similarity.item()

    return top_sentences

"""## pre_sum 함수"""

def pre_sum(news_list):

    # 전처리
    pre_news_list = []

    for news in tqdm(news_list):

        # 프로세스 0번 포토기사를 삭제
        if "포토" not in news['title']:

            # 프로레스 1번 %를 한국말 '퍼센트'로 바꿔준다
            news['pre_title'] = news['title'].replace('%', '퍼센트 ')
            news['pre_main_text'] = news['main_text'].replace('%', '퍼센트 ')

            # 프로세스 2번 특수문자를 제거한다
            news['pre_title'] = remove_special_characters(news['pre_title'])
            news['pre_main_text'] = remove_special_characters(news['pre_main_text'])

            # 프로세스 3번 제목과 유사한 문장을 본문에서 추출하여 추가한다
            news['summary'] = ' '.join(cal_similarity(news['title'], news['pre_main_text']))

            pre_news_list.append(news)

    return pre_news_list

"""### 저장하는 함수"""

def date_range(start, end):
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end-start).days+1)]
    return dates


def file_save(startdate_str: str, enddate_str: str, type=all):
    """
    startdate는 2023-07-09 형식으로
    """
    if type == "정치":
      for date in tqdm(date_range(startdate_str, enddate_str)):
          news_list = 멀티정치기사긁어오기(date)

          if news_list:
            pre_news_list = pre_sum(news_list)

            print(pre_news_list)

            news_to_json(pre_news_list, f'Politics_News_{date}.json')

    elif type == "사회":
      for date in tqdm(date_range(startdate_str, enddate_str)):
          news_list = 멀티사회기사긁어오기(date)

          if news_list:
            pre_news_list = pre_sum(news_list)

            print(pre_news_list)

            news_to_json(pre_news_list, f'Society_News_{date}.json')

    else:
      for date in tqdm(date_range(startdate_str, enddate_str)):
          news_list = 멀티정치기사긁어오기(date)

          if news_list:
            pre_news_list = pre_sum(news_list)

            print(pre_news_list)

            news_to_json(pre_news_list, f'Politics_News_{date}.json')
            print("정치 기사 저장 완료")

          news_list = 멀티사회기사긁어오기(date)

          if news_list:
            pre_news_list = pre_sum(news_list)

            print(pre_news_list)

            news_to_json(pre_news_list, f'Society_News_{date}.json')
            print("사회 기사 저장 완료")

file_save('시작 날짜', '종료 날짜', '정치 혹은 사회 기사')

