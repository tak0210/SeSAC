# 🔥 지역난방 열수요 예측 데이터 전처리 가이드

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [데이터 구조](#데이터-구조)
3. [전처리 과정](#전처리-과정)
4. [생성된 파생변수 상세 설명](#생성된-파생변수-상세-설명)
5. [사용법](#사용법)
6. [주의사항](#주의사항)

## 📊 프로젝트 개요

본 프로젝트는 **기상 데이터를 활용한 지역난방 열수요 예측**을 위한 데이터 전처리 파이프라인입니다. 

### 목표
- 기상변수(온도, 습도, 풍속, 강수량 등)를 활용하여 열수요 변화 패턴 분석
- 시계열 특성을 반영한 다양한 파생변수 생성
- 머신러닝 모델에 최적화된 형태로 데이터 전처리

### 데이터 특성
- **시간 단위**: 1시간 간격
- **대상 지역**: 한국지역난방공사 전국 19개 지사
- **기간**: 2021년(훈련), 2022년(테스트)
- **변수**: 기상 8개 + 열수요 1개

## 🗂️ 데이터 구조

### 원본 데이터 컬럼
| 컬럼명 | 설명 | 단위 | 특이사항 |
|--------|------|------|----------|
| `tm` | 시간 (YYYYMMDDHH) | - | 2021010101 형태 |
| `branch_id` | 지사 코드 | - | A, B, C, D, ... |
| `ta` | 기온 | °C | -99는 결측치 |
| `wd` | 풍향 | ° | -99는 결측치 |
| `ws` | 풍속 | m/s | -99는 결측치 |
| `rn_day` | 일강수량 | mm | -99는 결측치 |
| `rn_hr1` | 시간강수량 | mm | -99는 결측치 |
| `hm` | 습도 | % | -99는 결측치 |
| `si` | 일사량 | MJ/m² | 야간(18-08시) -99는 0처리 |
| `ta_chi` | 체감온도 | °C | -99는 결측치 |
| `heat_demand` | 열수요 | GJ | 예측 타겟 변수 |

## 🔧 전처리 과정

### 1단계: 데이터 로드 및 기본 처리
```python
# 컬럼명 정리 (train_heat. 접두사 제거)
# Unnamed: 0 컬럼 제거
# 기본 데이터 정보 확인
```

### 2단계: 결측치 처리
```python
# -99 값을 NaN으로 변환
# 일사량(si) 야간시간대(18-08시) 특별 처리: NaN → 0
# 지사별 시간순 선형보간
# Forward/Backward Fill로 끝단 결측치 처리
```

### 3단계: 시간 변수 생성
```python
# datetime 변환 및 기본 시간 변수 추출
# 순환형 인코딩 (sin/cos 변환)
# 계절/시간대 구분 변수
```

### 4단계: 기상 파생변수 생성
```python
# 물리적 의미 있는 기상 지표 생성
# 범주화 및 임계값 기반 변수
# Rolling 통계 및 지연 변수
```

### 5단계: 열수요 파생변수 생성
```python
# 열수요 패턴 분석 변수
# 효율성 및 민감도 지표
# 상대적 비교 변수
```

### 6단계: 상호작용 및 최종 처리
```python
# 변수 간 상호작용 특성
# 원핫 인코딩
# MinMax 스케일링
```

## 📈 생성된 파생변수 상세 설명

### 🕐 시간 관련 변수

#### 기본 시간 변수
- **`year, month, day, hour`**: 기본 시간 단위
- **`dayofweek`**: 요일 (0:월요일, 6:일요일)
- **`dayofyear`**: 연중 일수 (1-365)
- **`week`**: 연중 주차

#### 순환형 시간 변수 (Cyclical Encoding)
- **`hour_sin, hour_cos`**: 24시간 주기성 표현
  - 0시와 23시의 연속성 보장
- **`month_sin, month_cos`**: 12개월 주기성 표현
  - 12월과 1월의 연속성 보장
- **`dayofweek_sin, dayofweek_cos`**: 7일 주기성 표현
- **`dayofyear_sin, dayofyear_cos`**: 365일 주기성 표현

#### 계절 구분 변수
- **`season`**: 계절 구분 (0:겨울, 1:봄, 2:여름, 3:가을)
- **`heating_season`**: 난방시즌 (10-4월: 1, 그외: 0)
- **`peak_heating`**: 피크난방시즌 (12-2월: 1, 그외: 0)
- **`shoulder_season`**: 중간계절 (3-4월, 10-11월: 1, 그외: 0)

#### 시간대 구분 변수
- **`is_weekend`**: 주말 여부 (토일: 1, 평일: 0)
- **`is_work_hour`**: 근무시간 (9-18시: 1, 그외: 0)
- **`is_peak_morning`**: 오전 피크 (7-9시: 1, 그외: 0)
- **`is_peak_evening`**: 저녁 피크 (18-22시: 1, 그외: 0)
- **`is_night`**: 야간시간 (23-6시: 1, 그외: 0)

### 🌡️ 기상 파생변수

#### 온도 관련 지표
- **`HDD_18, HDD_20`**: 난방도일 (Heating Degree Day)
  - `HDD_18 = max(18 - 기온, 0)`
  - 에너지 효율 분야 표준 지표, 난방 필요량 추정
- **`CDD_26`**: 냉방도일 (Cooling Degree Day)
  - `CDD_26 = max(기온 - 26, 0)`

#### 체감 관련 지표
- **`wind_chill`**: 풍속을 고려한 체감온도
  - 실제 느끼는 온도, 난방 수요와 밀접한 관련
- **`discomfort_index`**: 불쾌지수 (온도 + 습도)
  - 실내 쾌적성 지표

#### 범주화 변수
- **`temp_category`**: 기온 범주 (0:매우추움 ~ 4:더움)
  - 구간: (-∞,0], (0,10], (10,20], (20,30], (30,∞)
- **`rain_intensity`**: 강수 강도 (0:무강수 ~ 4:매우강한비)
  - 구간: [0], (0,1], (1,5], (5,10], (10,∞)

#### 강수 관련 변수
- **`is_rainy`**: 강수 여부 (일강수량 > 0)
- **`is_heavy_rain`**: 강우 여부 (일강수량 > 10mm)

### 📊 Rolling 통계 변수

#### 기온 Rolling 통계 (6h, 12h, 24h, 48h, 168h)
- **`ta_mean_Xh`**: X시간 이동평균
- **`ta_std_Xh`**: X시간 이동표준편차 (변동성)
- **`ta_max_Xh`**: X시간 최고기온
- **`ta_min_Xh`**: X시간 최저기온

#### HDD Rolling 통계
- **`HDD_sum_Xh`**: X시간 누적 난방도일
  - 일정 기간의 누적 난방 필요량

#### 강수 Rolling 통계  
- **`rain_sum_Xh`**: X시간 누적 강수량

### ⏱️ 지연(Lag) 변수

#### 기온 지연 변수
- **`ta_lag_X`**: X시간 전 기온 (X = 1,2,3,6,12,24)
  - 기온 변화에 대한 열수요의 지연 반응 포착

#### 차분 및 변화율 변수
- **`ta_diff_1h`**: 1시간 기온 차분
- **`ta_diff_24h`**: 24시간 기온 차분 (전일 동시간 대비)
- **`ta_pct_change`**: 기온 변화율 (%)

### 🔥 열수요 관련 파생변수

#### 열수요 지연 변수
- **`demand_lag_X`**: X시간 전 열수요 (X = 1,2,3,6,12,24,48,168)
  - 열수요의 관성 및 시간 지연 효과

#### 열수요 Rolling 통계 (6h, 12h, 24h, 48h, 168h)
- **`demand_mean_Xh`**: X시간 평균 열수요
- **`demand_std_Xh`**: X시간 열수요 변동성
- **`demand_max_Xh, demand_min_Xh`**: 최대/최소 열수요

#### 열수요 변화 변수
- **`demand_diff_1h`**: 1시간 열수요 차분
- **`demand_diff_24h`**: 24시간 열수요 차분
- **`demand_pct_change_1h`**: 1시간 변화율
- **`demand_pct_change_24h`**: 24시간 변화율

#### 상대적 열수요 변수
- **`demand_vs_hourly_avg`**: 동일 시간대 평균 대비 비율
- **`demand_vs_weekly_avg`**: 동일 요일-시간 평균 대비 비율

#### 추세 변수
- **`demand_trend_24h`**: 24시간 선형 추세 (기울기)
- **`demand_trend_168h`**: 168시간(1주) 선형 추세

#### 효율성 지표
- **`heating_efficiency`**: 난방 효율성
  - `열수요 / (HDD_18 + ε)`
  - 단위 난방도일당 열수요량
- **`temp_sensitivity`**: 온도 민감도
  - `열수요 변화 / (기온 변화 + ε)`
  - 기온 변화에 대한 열수요 반응성

### 🔗 상호작용 변수

#### 기온 상호작용
- **`ta_hour_interaction`**: 기온 × 시간
  - 같은 온도라도 시간대별 열수요 패턴 차이
- **`ta_month_interaction`**: 기온 × 월
  - 계절별 온도 민감도 차이
- **`ta_weekend_interaction`**: 기온 × 주말

#### HDD 상호작용
- **`hdd_hour_interaction`**: 난방도일 × 시간
- **`hdd_weekend_interaction`**: 난방도일 × 주말

#### 기타 상호작용
- **`ta_humidity_interaction`**: 기온 × 습도
- **`wind_temp_interaction`**: 풍속 × 기온
- **`rain_season_interaction`**: 강수 × 계절

#### 지사별 상호작용
- **`branch_X_temp`**: 지사별 기온 반응성
- **`branch_X_hdd`**: 지사별 난방도일 반응성

## 🚀 사용법

### 1. 전처리 실행
```python
# 필요한 라이브러리 설치
pip install pandas numpy scikit-learn matplotlib seaborn

# 코드 실행
python heat_demand_preprocessing.py
```

### 2. 데이터 로드 경로 수정
```python
# 파일 경로를 실제 경로로 변경
train_df = load_data('your_path/train_heat.csv')
```

### 3. 결과 활용
```python
# 전처리된 데이터 사용
X_train = train_data.drop(columns=['heat_demand'])
y_train = train_data['heat_demand']

# 모델 훈련
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
```

## ⚠️ 주의사항

### 데이터 품질
1. **결측치**: -99 코드를 NaN으로 변환 후 선형보간
2. **일사량**: 야간시간(18-08시) -99는 0으로 처리
3. **시간순 정렬**: 지연 변수 생성 전 반드시 시간순 정렬 필요

### 메모리 사용량
- **Rolling 통계**: 큰 윈도우(168h) 사용시 메모리 사용량 증가
- **지연 변수**: 많은 지연 변수 생성시 컬럼 수 급증
- **권장**: 필요한 변수만 선택적으로 생성

### 시계열 특성
1. **Data Leakage 방지**: 미래 정보 사용 금지
2. **시간순 분할**: 무작위 분할 대신 시간순 분할 사용
3. **검증 전략**: Time Series Cross-Validation 권장

### 모델링 고려사항
1. **Feature Selection**: 생성된 변수가 많으므로 특성 선택 필요
2. **Regularization**: 과적합 방지를 위한 정규화 적용
3. **앙상블**: 여러 모델의 조합으로 성능 향상

## 📊 성능 개선 팁 (그냥 참고용)

### Feature Engineering
1. **도메인 지식 활용**: 난방 전문가 의견 반영
2. **계절성 강화**: 지역별 기후 특성 반영

### 모델링 전략
1. **시계열 모델**: ARIMA, Prophet, LSTM 등 고려
2. **앙상블**: LightGBM + 시계열 모델 조합
3. **Post-processing**: 물리적 제약조건 반영

## 📞 추가 피드백 및 수정사항

파생변수 추가하거나 수정할 내용이 있다면 아래에 작성해주세요.


| LSTM 변수                     | 중요도 |
| ----------------------------- | ------: |
| `train_heat.heat_demand_ma24` | 39.4827 |
| `hour`                        |  5.7657 |
| `train_heat.ta_chi`           |  2.4199 |
| `train_heat.heat_demand_diff` |  0.9041 |
| `train_heat.si`               |  0.8527 |
| `train_heat.ta`               |  0.8218 |
| `Unnamed: 0`                  |  0.2417 |
| `is_weekend`                  |  0.2073 |
| `month`                       |  0.1875 |
| `train_heat.ws`               |  0.1611 |
| `train_heat.wd`               |  0.0788 |
| `train_heat.rn_day`           |  0.0491 |
| `train_heat.rn_hr1`           |  0.0276 |
| `train_heat.hm`               | –0.2820 |



| GRU 변수                      | 중요도 |
| ----------------------------- | ------: |
| `train_heat.heat_demand_ma24` | 43.3630 |
| `hour`                        |  3.6996 |
| `train_heat.heat_demand_diff` |  2.5452 |
| `train_heat.ta_chi`           |  2.2803 |
| `train_heat.si`               |  1.6223 |
| `train_heat.ta`               |  0.9232 |
| `train_heat.hm`               |  0.3408 |
| `is_weekend`                  |  0.2602 |
| `train_heat.ws`               |  0.2105 |
| `Unnamed: 0`                  |  0.2031 |
| `month`                       |  0.0820 |
| `train_heat.wd`               |  0.0426 |
| `train_heat.rn_hr1`           |  0.0173 |
| `train_heat.rn_day`           | –0.0022 |



---
**마지막 업데이트**: 2024년 6월 08일
**버전**: 1.0.0
