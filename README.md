# SeSAC
공모전 파이팅

## 6/1
### Baseline 폴더 추가
  1. 시계열 데이터 파생 및 결측치 처리
  2. Train / Test Dataset 분리 후 MICE 진행
     -. 각 데이터셋 추출 완료 (용량 제한으로 업로드 불가)
=> MICE 미사용으로 결정, 해당 전처리 폴더 삭제 예정

## 6/5 
### 1차 모델링 폴더 추가

**[모델링]**
- 지사 : A, B, D
- Train : 21년 (Val 8:2), Test: 22년
- 파생변수 추가, 변수 조절해서 해보기
- 보간법 : 선형 보간, ffill, bfill 등
1. **일반 시계열 (하는 사람이 잘 List-up) Prophet, DLinear - 박지수**
2. **딥러닝 (CNN 기반, RNN 기반, Transformer 기반) - 김가영**
3. **딥러닝 앙상블 (CNN + Transformer) - 신윤식**
4. **지사 반영_임베딩 (TFT, DeepAR_Pytorch Forecasting) - 황인탁**

**학습 결과**
| 모델명 | RMSE값 | 사용 메모리 | 학습 시간 | 검증 시간 |
- LSTM 모델 RMSE(파생변수 포함): 13.0315

## 6/7
### 전처리 폴더 추가

**파생변수 List-up**
