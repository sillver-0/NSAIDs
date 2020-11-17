# NSAIDs
Word2Vec과 LSTM을 활용한 비스테로이드성 소염 진통제의 약물 부작용에 대한 사용자 리뷰 분석

매년 빠르게 증가하는 골관절염 환자 수에 따라 자연스레 관심이 커지고 있는 비스테로이드성 소염 진통제 (NSAIDs) 약물에 대한 사용자 리뷰를 분석하고자 하였다.
평점이 포함된 약물 리뷰 38,022개를 훈련 데이터로 활용하였다.
drugs.com, askapatient.com 등 5개의 약물 리뷰 사이트에서 수집한 Celecoxib 약물 리뷰 604개, Naproxen 리뷰 861개, 그리고 ibuprofen 리뷰 285개를 수집하여 약물 리뷰 총 39,772건을 활용하여 연구를 진행하였다.
사용자가 부여한 평점을 기준으로 레이블링한 기존 LSTM 모델과 차별성을 두기 위해 부작용 데이터를 활용하였다.
Word2Vec을 이용하여 부작용 단어들과의 유사 단어를 수집하였다.
이를 바탕으로 약물 리뷰를 다시 레이블링하여 LSTM 모델을 생성하였다.
기존 모델과 비교하여 Word2Vec을 이용해 부작용 데이터를 활용한 모델의 정확도가 더 높았다.
즉, 부작용 데이터가 사용자 리뷰를 분석하는 데 큰 영향을 미치는 것을 확인하였다.



Test_raw_3_9.csv: 해커톤 데이터를 활용한 학습 데이터. 평점 0-3의 리뷰를 부정 리뷰로, 9-10의 리뷰를 긍정 리뷰로 활용하였다.

celecoxib.csv, ibuprofen.csv, naproxen.csv: drugs.com, askapatient.com 등 5개의 약물 리뷰 사이트에서 수집한 약물 리뷰이다.

side_effect.txt: drugs.com에서 제공한 Celecoxib, Naproxen, Ibuprofen의 부작용 데이터이다.

lstm_0420.py: LSTM을 활용한 예측 모델. Word2Vec을 활용하여 레이블링을 다시 진행하였다.
