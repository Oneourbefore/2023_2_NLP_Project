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
- [KPF-SBERT](https://huggingface.co/bongsoo/kpf-sbert-v1.1)로 임베딩
- 클러스터별 키워드 추출
- v1.0(2023-08-20)
    - 기사 제목에서 기사 제목과 유사도가 높은 키워드를 추출
    - 키워드의 다양성을 높이기 위해 MMR (Maximum Marginal Relevance) 기법을 사용
- v1.1(2023-08-21)
    - 불용어 목록 수정 (안해, 않았어야 등)
- v1.2(2023-08-22)
    - 불용어 목록 수정 (vs, 안돼 등)
    - 에러 수정
- v1.3(2023-08-23)
    - 불용어 목록 수정 (없어 등)
    - 숫자로만 이루어진 키워드는 제외

### Choose best title per cluster
- 클러스터별 키워드가 가장 많이 포함된 기사의 제목을 클러스터별 대표 기사 제목으로 채택
- 이후 연결될 클러스터들끼리 정렬하여 가장 최근 클러스터의 대표 기사 제목을 사건의 소제목으로 활용함

## Connecting clusters & Issuing a new event
- [KPF-SBERT](https://huggingface.co/bongsoo/kpf-sbert-v1.1)로 임베딩
- 기준 날짜로부터 가장 최근의 20개 클러스터들의 키워드와 비교해서 코사인 유사도가 임계치를 넘으면 과거의 클러스터와 같은 사건이라고 판단하고, 동일 사건으로 묶이도록 함
- 임계치를 넘지 못할 경우 새로운 사건으로 판단하여 발행함

  ![connecting_cluster.png]((https://github.com/WordWarrior-NLP/Kordata-nlp-models/blob/main/connecting_cluster.png))
* 오늘의 클러스터 (nc_id 71~75)를 과거의 클러스터 (nc_id 51~70)

### Naming a new event
- **Branch Entropy** 이용
- 클러스터로 묶인 기사별 요약에서 명사만 추출하여 사건 이름으로 작성
- Branch Entropy가 낮은 순으로 3개의 단어를 골라 사건 이름으로 결정

## Named Entity Recognition
- model: KPF-BERT-NER,Tokenizer: KPFbert
- NER 다음 BIO태깅을 사용하여 I-name + I-position인 이름+직책, 기관(OGG)를 사용
- 자카드 유사도를 사용하여 임계치를 0.6<i <1.0 으로 설정하고 유사한 딕셔너리 추출
- 딕셔너리의 쌍이 맞는 경우 추출 e.g. 문 대통령 - 문재인 대통령 , 문재인 대통령- 문 대통령 -> 문재인 대통령 - 문대통령 
- 기사의 특성상 풀네임이 먼저 언급되고 약어가 사용됨 더 긴 글자가 대표하는 단어가 된다고 판단 (최장일치법) e.g. 윤석열 대통령 - > 윤 대통령  
- 토크나이저로 인해서 NER의 결과가 잘못된 것 (e.g. 재명더불어민주당)은 summary에서 실제로 그 개체가 있는지 확인 후 없을 경우 동의어에서 제거
- 이렇게 전처리를 진행 한 후 날짜별로 클러스별 상위 주체 5개를 뽑은 후, 동의어를 main_word에 대치 후 형식을 맞춰 (word, nc_id, label, `desc`, nid, datetime, main_word) DB에 저장

