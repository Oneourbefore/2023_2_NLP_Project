# Kordata-nlp-models
NLP models for Clustering, Extracting keywords, Connecting clusters, NER, and Sentiment classification

---
### Dataset
#### NaverNewsScraping
- 매일 오전 10시에 전날의 뉴스를 수집.
- 네이버 뉴스의 정치 섹션 뉴스를 수집.

#### PreSum
- v1.0:
    - 포토 기사는 수집하지 않는다.
    - **summary**: 기사 본문 중 가장 기사 제목과 유사한 세 문장을 추출하여 요약. SentenceTransformer로 벡터화, cosine similarity 계산
    - **pre_main_text**: 특수문자를 제거하고 '%'는 '퍼센트'로 바꿔준다.
- v1.1 (2023/07/30): reporter, photo 추가
    - reporter: 기자 이름
    - photo: 사진 유무 (True or False)
    - main_text: 기사 부제가 중복되게 추가되지 않도록 수정
- v1.2 (2023/08/07): 사회면 기사 수집
    - 멀티사회기사긁어오기

#### SaveToDB
- DB에 저장시 중복 저장을 피하기 위해 INSERT IGNORE 사용.
- photo 컬럼은 8월 10일부터 available.

## Clustering
- KPF-SBERT 사용
- 매일 생성된 
- HDBScan과 umap 차원 축소를 통해

### Extracting keywords per cluster
- KPF-SBERT(https://huggingface.co/bongsoo/kpf-sbert-v1.1).로 임베딩
- 클러스터별 키워드 추출
- 기사 제목에서 기사 제목과 유사도가 높은 키워드를 추출
- 키워드의 다양성을 높이기 위해 MMR 기법을 사용

## Connecting clusters & Issuing a new event
- KPF-SBERT(https://huggingface.co/bongsoo/kpf-sbert-v1.1).로 임베딩
- 기준 날짜로부터 과거 5일 동안 생성된 클러스터들의 키워드와 비교해서 코사인 유사도가 임계치를 넘으면 과거의 클러스터와 같은 사건으로 묶이도록 함
- 임계치를 넘지 못할 경우 새로운 사건으로 판단하여 발행함

### Naming a new event
- **Branch Entropy** 이용
- 클러스터로 묶인 기사별 요약에서 명사만 추출하여 사건 이름으로 작성
- Branch Entropy가 낮은 순으로 3개의 단어를 골라 사건 이름으로 결정

## 
