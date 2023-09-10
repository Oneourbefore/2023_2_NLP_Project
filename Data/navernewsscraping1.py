# -*- coding: utf-8 -*-
"""NaverNewsScraping1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jHGfYSOfR388SbyDYtv1K8IsMN7BK3pe
"""

# -*- coding: utf-8 -*-
"""NaverNewsScraping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11pWhFxMsul2aVggSPczjOPhEyOZjOk43
"""

from requests import request
import requests
from requests.compat import *
from bs4 import BeautifulSoup
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import timeit

urls = {
    'home': 'https://media.naver.com/newsflash/',
    'sum': 'https://tts.news.naver.com/article/'
}

press_id = {
    '경향신문': '032',
    '국민일보': '005',
    '동아일보': '020',
    '문화일보': '021',
    '서울신문': '081',
    '세계일보': '022',
    '조선일보': '023',
    '중앙일보': '025',
    '한겨레': '028',
    '한국일보': '469'
}

def 언론사별정치기사긁어오기(upress: str, udate: str):
    """
    udate는 2023-07-09 형식으로
    upress는 press_id 참조
    """

    news_list = []
    url = urls['home'] + press_id[upress] + '/Politics'

    udate = datetime.strptime(udate, "%Y-%m-%d").date()

    resp = requests.get(url)
    raw_news_list = resp.json()['list']

    timestamp = raw_news_list[-1]['datetime'] / 1000
    date_obj = datetime.fromtimestamp(timestamp)
    last_news_issue_date = date_obj.date()

    while True:
        for i, raw_news in enumerate(raw_news_list):
            timestamp = raw_news['datetime'] / 1000
            date_obj = datetime.fromtimestamp(timestamp)
            issue_date = date_obj.date()

            if issue_date == udate: # 해당되는 날짜의 기사만 받아오기


                news = {
                        'nid': raw_news['id'],
                        'pid': raw_news['groupId'],
                        'title': raw_news['title'],
                        'linkUrl': raw_news['linkUrl'],
                        'datetime': raw_news['datetime'],
                        'strtime': raw_news['serviceMonthDayTime'],
                }

                resp = requests.get(news['linkUrl'])
                soup = BeautifulSoup(resp.text, 'lxml')  # lxml 파서를 사용하여 BeautifulSoup 객체 생성

                news['main_text'] = soup.select_one('#dic_area').get_text(strip=True)

                #sen_list = split_sentences(news['main_text'])
                #news['articleSummary'] = summarize_article(sen_list)

                news_list.append(news)

        # 마지막 기사의 datetime = input datetime이면 다음 목록의 기사들도 호출
        if last_news_issue_date >= udate:
            resp = requests.get(url + f'?before={raw_news_list[-1]["serviceTimeForMoreApi"]}')
            raw_news_list = resp.json()['list']

            if raw_news_list:
                timestamp = raw_news_list[-1]['datetime'] / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                last_news_issue_date = date_obj.date()
                continue

        break

    return news_list

def 언론사별사회기사긁어오기(upress: str, udate: str):
    """
    udate는 2023-07-09 형식으로
    upress는 press_id 참조
    """

    news_list = []
    url = urls['home'] + press_id[upress] + '/Society'

    udate = datetime.strptime(udate, "%Y-%m-%d").date()

    resp = requests.get(url)
    raw_news_list = resp.json()['list']

    timestamp = raw_news_list[-1]['datetime'] / 1000
    date_obj = datetime.fromtimestamp(timestamp)
    last_news_issue_date = date_obj.date()

    while True:
        for i, raw_news in enumerate(raw_news_list):
            timestamp = raw_news['datetime'] / 1000
            date_obj = datetime.fromtimestamp(timestamp)
            issue_date = date_obj.date()

            if issue_date == udate: # 해당되는 날짜의 기사만 받아오기


                news = {
                        'nid': raw_news['id'],
                        'pid': raw_news['groupId'],
                        'title': raw_news['title'],
                        'linkUrl': raw_news['linkUrl'],
                        'datetime': raw_news['datetime'],
                        'strtime': raw_news['serviceMonthDayTime'],
                }

                resp = requests.get(news['linkUrl'])
                soup = BeautifulSoup(resp.text, 'lxml')  # lxml 파서를 사용하여 BeautifulSoup 객체 생성

                news['main_text'] = soup.select_one('#dic_area').get_text(strip=True)

                #sen_list = split_sentences(news['main_text'])
                #news['articleSummary'] = summarize_article(sen_list)

                news_list.append(news)

        # 마지막 기사의 datetime = input datetime이면 다음 목록의 기사들도 호출
        if last_news_issue_date >= udate:
            resp = requests.get(url + f'?before={raw_news_list[-1]["serviceTimeForMoreApi"]}')
            raw_news_list = resp.json()['list']

            if raw_news_list:
                timestamp = raw_news_list[-1]['datetime'] / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                last_news_issue_date = date_obj.date()
                continue

        break

    return news_list

def fetch_url(url):
    response = requests.get(url)
    return response.text

def parse_html(html):
    soup = BeautifulSoup(html, 'lxml')
    photo = False
    end_photo_org_spans = soup.find_all('span', class_='end_photo_org')
    if end_photo_org_spans:
      photo = True
      for end_photo in end_photo_org_spans:
              end_photo.decompose()

    main_text = soup.select_one('#dic_area').get_text(strip=True)
    return main_text, photo

def reporter(html):
    soup = BeautifulSoup(html, 'lxml')

    reporter_object = soup.find_all('em', class_= 'media_end_head_journalist_name')
    if reporter_object:
      for reporter in reporter_object:
        name = reporter.get_text(strip=True)
        reporter_name =  re.sub(r'\s+기자$', '', name)
    else:
        reporter_name = 'None'
    return reporter_name

def news_to_json(news_list: list, filename: str):
    filepath = '/content/drive/My Drive/Kordata/dataset/' + filename  # Change the path according to your desired location in Drive
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(news_list, json_file, ensure_ascii=False)

def 멀티언론사별정치기사긁어오기(upress: str, udate: str):
    """
    udate는 2023-07-09 형식으로
    upress는 press_id 참조
    """

    news_list = []
    url = urls['home'] + press_id[upress] + '/Politics'

    udate = datetime.strptime(udate, "%Y-%m-%d").date()

    resp = requests.get(url)
    raw_news_list = resp.json()['list']

    timestamp = raw_news_list[-1]['datetime'] / 1000
    date_obj = datetime.fromtimestamp(timestamp)
    last_news_issue_date = date_obj.date()

    news_urls = []

    while True:

        for i, raw_news in enumerate(raw_news_list):
            timestamp = raw_news['datetime'] / 1000
            date_obj = datetime.fromtimestamp(timestamp)
            issue_date = date_obj.date()

            if issue_date == udate: # 해당되는 날짜의 기사만 받아오기

                news = {
                        'nid': raw_news['id'],
                        'pid': raw_news['groupId'],
                        'title': raw_news['title'],
                        'linkUrl': raw_news['linkUrl'],
                        'datetime': raw_news['datetime'],
                        'strtime': raw_news['serviceMonthDayTime'],
                }

                news_urls.append(raw_news['linkUrl'])
                news_list.append(news)


        # 마지막 기사의 datetime = input datetime이면 다음 목록의 기사들도 호출
        if last_news_issue_date >= udate:
            resp = requests.get(url + f'?before={raw_news_list[-1]["serviceTimeForMoreApi"]}')
            raw_news_list = resp.json()['list']

            if raw_news_list:
                timestamp = raw_news_list[-1]['datetime'] / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                last_news_issue_date = date_obj.date()
                continue

        break

    with ThreadPoolExecutor(max_workers=5) as executor:
        htmls = executor.map(fetch_url, news_urls)
        for html, news in zip(htmls, news_list):
            news['main_text'], news['photo'] = parse_html(html)
            news['reporter'] = reporter(html)
            #news['articleSummary'] = summarize_article(split_sentences(news['main_text']))

    return news_list

def 정치기사긁어오기(udate: str):
    """
    해당되는 날짜에 모든 종합지의 기사 긁어오기
    udate는 2023-07-09 형식으로
    """
    news_list = []

    for press in press_id.keys():
        news_list.extend(언론사별정치기사긁어오기(press, udate))

    return news_list

def 여러날정치기사긁어오기(udate: str, num_days: int):
    """
    udate는 2023-07-09 형식으로
    기준 날짜를 가장 최신 날짜로 해서 그 날을 기준으로 여러날을 긁어옴
    num_days는 정수형
    """

    date = udate[-2:]
    int_date = int(date)

    news_list = []

    for date in range(int_date-num_days+1, int_date+1):
        date = str(date)
        news_list.extend(정치기사긁어오기(f'{udate[:-2]+date}'))

    return news_list

def 멀티언론사별사회기사긁어오기(upress: str, udate: str):
    """
    udate는 2023-07-09 형식으로
    upress는 press_id 참조
    """

    news_list = []
    url = urls['home'] + press_id[upress] + '/Society'

    udate = datetime.strptime(udate, "%Y-%m-%d").date()

    resp = requests.get(url)
    raw_news_list = resp.json()['list']

    timestamp = raw_news_list[-1]['datetime'] / 1000
    date_obj = datetime.fromtimestamp(timestamp)
    last_news_issue_date = date_obj.date()

    news_urls = []

    while True:

        for i, raw_news in enumerate(raw_news_list):
            timestamp = raw_news['datetime'] / 1000
            date_obj = datetime.fromtimestamp(timestamp)
            issue_date = date_obj.date()

            if issue_date == udate: # 해당되는 날짜의 기사만 받아오기

                news = {
                        'nid': raw_news['id'],
                        'pid': raw_news['groupId'],
                        'title': raw_news['title'],
                        'linkUrl': raw_news['linkUrl'],
                        'datetime': raw_news['datetime'],
                        'strtime': raw_news['serviceMonthDayTime'],
                }

                news_urls.append(raw_news['linkUrl'])
                news_list.append(news)


        # 마지막 기사의 datetime = input datetime이면 다음 목록의 기사들도 호출
        if last_news_issue_date >= udate:
            resp = requests.get(url + f'?before={raw_news_list[-1]["serviceTimeForMoreApi"]}')
            raw_news_list = resp.json()['list']

            if raw_news_list:
                timestamp = raw_news_list[-1]['datetime'] / 1000
                date_obj = datetime.fromtimestamp(timestamp)
                last_news_issue_date = date_obj.date()
                continue

        break

    with ThreadPoolExecutor(max_workers=5) as executor:
        htmls = executor.map(fetch_url, news_urls)
        for html, news in zip(htmls, news_list):
            news['main_text'], news['photo'] = parse_html(html)
            news['reporter'] = reporter(html)
            #news['articleSummary'] = summarize_article(split_sentences(news['main_text']))

    return news_list

def 사회기사긁어오기(udate: str):
    """
    해당되는 날짜에 모든 종합지의 기사 긁어오기
    udate는 2023-07-09 형식으로
    """
    news_list = []

    for press in press_id.keys():
        news_list.extend(언론사별사회기사긁어오기(press, udate))

    return news_list

def 여러날사회기사긁어오기(udate: str, num_days: int):
    """
    udate는 2023-07-09 형식으로
    기준 날짜를 가장 최신 날짜로 해서 그 날을 기준으로 여러날을 긁어옴
    num_days는 정수형
    """

    date = udate[-2:]
    int_date = int(date)

    news_list = []

    for date in range(int_date-num_days+1, int_date+1):
        date = str(date)
        news_list.extend(사회기사긁어오기(f'{udate[:-2]+date}'))

    return news_list

def 멀티정치기사긁어오기(udate: str):
    """
    해당되는 날짜에 모든 종합지의 기사 긁어오기
    udate는 2023-07-09 형식으로
    """
    news_list = []

    for press in press_id.keys():
        news_list.extend(멀티언론사별정치기사긁어오기(press, udate))

    return news_list


def 멀티여러날정치기사긁어오기(udate: str, num_days: int):
    """
    udate는 2023-07-09 형식으로
    기준 날짜를 가장 최신 날짜로 해서 그 날을 기준으로 여러날을 긁어옴
    num_days는 정수형
    """

    date = udate[-2:]
    int_date = int(date)

    news_list = []

    for date in range(int_date-num_days+1, int_date+1):
        date = str(date)
        news_list.extend(멀티정치기사긁어오기(f'{udate[:-2]+date}'))

    return news_list

def 멀티사회기사긁어오기(udate: str):
    """
    해당되는 날짜에 모든 종합지의 기사 긁어오기
    udate는 2023-07-09 형식으로
    """
    news_list = []

    for press in press_id.keys():
        news_list.extend(멀티언론사별사회기사긁어오기(press, udate))

    return news_list


def 멀티여러날사회기사긁어오기(udate: str, num_days: int):
    """
    udate는 2023-07-09 형식으로
    기준 날짜를 가장 최신 날짜로 해서 그 날을 기준으로 여러날을 긁어옴
    num_days는 정수형
    """

    date = udate[-2:]
    int_date = int(date)

    news_list = []

    for date in range(int_date-num_days+1, int_date+1):
        date = str(date)
        news_list.extend(멀티사회기사긁어오기(f'{udate[:-2]+date}'))

    return news_list

# 멀티언론사별정치기사긁어오기('경향신문', '2023-07-24')

# 수정 전

# if __name__ == '__main__':
#     udate = '2023-07-23'
#     time_taken = timeit.timeit(lambda: 언론사별정치기사긁어오기('경향신문', udate), number=1)
#     print(f"실행 시간: {time_taken} 초")

# if __name__ == '__main__':
#     udate = '2023-07-23'
#     time_taken = timeit.timeit(lambda: 정치기사긁어오기(udate), number=1)
#     print(f"실행 시간: {time_taken} 초")

# if __name__ == '__main__':
#     udate = '2023-07-23'
#     time_taken = timeit.timeit(lambda: 여러날정치기사긁어오기(udate, 5), number=1)
#     print(f"실행 시간: {time_taken}초, 약 {time_taken / 60} 분")

# if __name__ == '__main__':
#     udate = '2023-07-23'
#     time_taken = timeit.timeit(lambda: 여러날정치기사긁어오기(udate, 7), number=1)
#     print(f"실행 시간: {time_taken}초, 약 {time_taken / 60} 분")

# 수정 후

if __name__ == '__main__':
    udate = '2023-07-23'
    time_taken = timeit.timeit(lambda: 멀티언론사별정치기사긁어오기('경향신문', udate), number=1)
    print(f"실행 시간: {time_taken} 초")

    time_taken = timeit.timeit(lambda: 멀티정치기사긁어오기(udate), number=1)
    print(f"실행 시간: {time_taken} 초")

    time_taken = timeit.timeit(lambda: 멀티여러날정치기사긁어오기(udate, 5), number=1)
    print(f"실행 시간: {time_taken}초, 약 {time_taken / 60} 분")

    time_taken = timeit.timeit(lambda: 멀티여러날정치기사긁어오기(udate, 7), number=1)
    print(f"실행 시간: {time_taken}초, 약 {time_taken / 60} 분")

if __name__ == '__main__':
    udate = '2023-08-04'
    print(멀티언론사별사회기사긁어오기('경향신문', udate))

