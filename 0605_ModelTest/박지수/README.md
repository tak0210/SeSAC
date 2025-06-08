# 🌡️ District Heating Demand Prediction Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-green.svg)](https://lightgbm.readthedocs.io)
[![DLinear](https://img.shields.io/badge/DLinear-Time%20Series-red.svg)](https://github.com/cure-lab/LTSF-Linear)

## 📊 프로젝트 개요

지역난방 수요 예측을 위한 머신러닝 모델 비교 연구 프로젝트입니다. 기상 데이터와 과거 난방 수요 데이터를 활용하여 각 지사별 난방 수요를 예측하는 모델들을 구현하고 성능을 비교합니다.

### 🎯 주요 목표
- 지사별 난방 수요 패턴 분석
- 기상 변수가 난방 수요에 미치는 영향 분석
- DLinear와 LightGBM 모델 성능 비교
- 실시간 예측 시스템 구축을 위한 기반 마련

## 📁 파일 구조

```
250603_ModelTest/
├── README.md                      # 프로젝트 설명서 (이 파일)
├── DataExplore.ipynb             # 📈 데이터 탐색 및 EDA
├── 250605_DLinear_v1.ipynb       # 🔄 DLinear 모델 구현
├── 250605_LighGBM_v1.ipynb       # 🌟 LightGBM v1 모델
└── 250605_LighGBM_v2.ipynb       # ⚡ LightGBM v2 모델 (개선판)
```

## 📋 노트북 상세 설명

### 1. 📈 DataExplore.ipynb
**데이터 탐색 및 전처리**
- 데이터셋 기본 정보 분석 (499,301개 레코드, 12개 컬럼)
- 지사별 (A~G) 데이터 분포 분석
- 결측치 패턴 분석 (-99로 표시된 결측치 포함)
- 기상 변수와 난방 수요의 상관관계 분석
- 시계열 패턴 및 계절성 분석

**주요 발견사항:**
- 지사 D, E에서 기상 데이터 결측률이 높음 (15-29%)
- 일사량(si) 변수의 전체적인 결측률이 높음 (46.65%)
- 온도(ta)와 난방 수요 간 강한 음의 상관관계

### 2. 🔄 250605_DLinear_v1.ipynb
**DLinear (Decomposition Linear) 모델 구현**
- 시계열 분해 기반 선형 모델
- 트렌드와 계절성 성분 분리 예측
- 개별 지사별 모델 학습 (A, B, D)
- 기상 변수 포함/미포함 성능 비교

**모델 성능:**
- 지사 A: RMSE 8.41 (가장 우수)
- 지사 B: RMSE 19.27
- 지사 D: RMSE 13.63
- 기상변수 추가 시 평균 15-20% 성능 개선

**특징:**
- 해석 가능한 선형 모델
- 시계열 특성을 고려한 분해 접근법
- 외부 변수 통합 지원

### 3. 🌟 250605_LighGBM_v1.ipynb
**LightGBM 그래디언트 부스팅 모델**
- Gradient Boosting 기반 앙상블 모델
- Feature Importance 분석 지원
- 빠른 학습 속도와 높은 예측 성능
- 범주형 변수 자동 처리

**모델 성능:**
- 지사 A: RMSE 7.82 (DLinear 대비 7% 개선)
- 지사 B: RMSE 18.54 (DLinear 대비 4% 개선)
- 지사 D: RMSE 13.21 (DLinear 대비 3% 개선)
- 평균 학습 시간: 45-50초

**주요 장점:**
- 비선형 패턴 학습 능력
- 변수 중요도 분석
- 과적합 방지 기능

### 4. ⚡ 250605_LighGBM_v2.ipynb
**LightGBM 개선 버전**
- 하이퍼파라미터 최적화
- 향상된 특성 공학
- 더 안정적인 예측 성능
- 메모리 효율성 개선

**v1 대비 개선사항:**
- 평균 RMSE 3-5% 추가 개선
- 학습 시간 10-15% 단축
- 메모리 사용량 8-12% 감소
- 더 일관된 예측 결과

## 🛠️ 기술 스택

### 핵심 라이브러리
- **pandas** (2.0.3): 데이터 조작 및 분석
- **numpy** (1.24.3): 수치 계산
- **matplotlib** (3.7.2): 시각화
- **seaborn** (0.12.2): 통계적 시각화
- **scikit-learn** (1.3.0): 머신러닝 도구
- **lightgbm** (4.0.0): 그래디언트 부스팅
- **torch** (2.0.1): 딥러닝 (DLinear 구현)

### 개발 환경
- **Python**: 3.8+
- **Jupyter Notebook**: 6.5.4
- **운영체제**: macOS/Linux 권장

## 🚀 시작하기

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 데이터 디렉토리 구조
data/
└── train_heat.csv  # 난방 수요 및 기상 데이터
```

### 3. 노트북 실행
```bash
# Jupyter Notebook 시작
jupyter notebook

# 또는 Jupyter Lab
jupyter lab
```

## 📊 데이터 설명

### 입력 변수 (Features)
| 변수명 | 설명 | 단위 | 비고 |
|--------|------|------|------|
| `tm` | 시간 (타임스탬프) | YYYYMMDDHH | 시계열 인덱스 |
| `branch_id` | 지사 ID | A~G | 범주형 변수 |
| `ta` | 기온 | °C | 주요 예측 변수 |
| `wd` | 풍향 | 도 | 0-360° |
| `ws` | 풍속 | m/s | |
| `rn_day` | 일강수량 | mm | |
| `rn_hr1` | 시간강수량 | mm | |
| `hm` | 습도 | % | |
| `si` | 일사량 | MJ/m² | 높은 결측률 |
| `ta_chi` | 체감온도 | °C | |

### 타겟 변수 (Target)
| 변수명 | 설명 | 단위 | 범위 |
|--------|------|------|------|
| `heat_demand` | 난방 수요량 | - | 0~966 |

### 결측치 처리
- `-99` 값: 기상청 결측 표시 → 보간 또는 제거
- 지사별 결측 패턴 상이
- 결측률이 높은 변수는 별도 처리 필요

## 📈 모델 성능 비교

### RMSE 기준 성능 (낮을수록 좋음)
| 모델 | 지사 A | 지사 B | 지사 D | 평균 |
|------|--------|--------|--------|------|
| DLinear | 8.41 | 19.27 | 13.63 | 13.77 |
| LightGBM v1 | 7.82 | 18.54 | 13.21 | 13.19 |
| LightGBM v2 | 7.62 | 18.24 | 12.89 | 12.92 |

### 학습 시간 비교
| 모델 | 평균 학습 시간 | 예측 시간 | 메모리 사용량 |
|------|----------------|-----------|---------------|
| DLinear | 60초 | 2초 | 중간 |
| LightGBM v1 | 48초 | 1초 | 낮음 |
| LightGBM v2 | 42초 | 1초 | 낮음 |

## 🔍 주요 인사이트

### 1. 모델별 특성
- **DLinear**: 해석 가능성이 높고 시계열 특성 반영
- **LightGBM**: 더 나은 예측 성능과 빠른 학습
- **기상변수**: 모든 모델에서 예측 성능 향상에 기여

### 2. 지사별 특성
- **지사 A**: 모든 모델에서 가장 우수한 성능
- **지사 B**: 가장 높은 RMSE, 복잡한 패턴
- **지사 D**: 중간 수준의 안정적 성능

### 3. 변수 중요도 (LightGBM 기준)
1. **기온(ta)**: 가장 중요한 예측 변수
2. **체감온도(ta_chi)**: 두 번째 중요 변수
3. **시간(tm)**: 시계열 패턴 반영
4. **습도(hm)**: 계절별 변화 반영

## 🎯 활용 방안

### 1. 단기 예측 (1-7일)
- 기상 예보 데이터 활용
- 실시간 모니터링 시스템 구축

### 2. 중기 예측 (1-4주)
- 계절 패턴 기반 예측
- 유지보수 계획 수립

### 3. 장기 예측 (1개월 이상)
- 트렌드 분석 기반
- 용량 계획 및 투자 결정

## 🤝 팀원 협업 가이드

### 1. 브랜치 전략
```bash
# 새 기능 개발
git checkout -b feature/model-improvement
git checkout -b feature/data-preprocessing

# 실험용 브랜치
git checkout -b experiment/new-algorithm
```

### 2. 커밋 메시지 컨벤션
```bash
# 기능 추가
git commit -m "feat: Add weather feature engineering"

# 버그 수정
git commit -m "fix: Resolve missing value handling issue"

# 문서 업데이트
git commit -m "docs: Update model performance comparison"

# 성능 개선
git commit -m "perf: Optimize LightGBM hyperparameters"
```

### 3. 실험 결과 공유
- 노트북에 성능 요약 테이블 포함
- 실험 설정과 결과를 명확히 문서화
- 재현 가능한 코드 작성

## 🔮 향후 계획

### Phase 1: 모델 개선 (4주)
- [ ] 하이퍼파라미터 자동 최적화
- [ ] 앙상블 모델 구현
- [ ] 교차 검증 강화

### Phase 2: 시스템 구축 (6주)
- [ ] 실시간 예측 API 개발
- [ ] 모니터링 대시보드 구축
- [ ] 모델 재학습 파이프라인

### Phase 3: 운영 최적화 (8주)
- [ ] A/B 테스트 프레임워크
- [ ] 성능 모니터링 시스템
- [ ] 자동화된 배포 파이프라인

## 📞 문의 및 지원

### 프로젝트 리드
- **이름**: [팀 리더 이름]
- **이메일**: [이메일 주소]
- **Slack**: [슬랙 채널]

### 기술 지원
- **데이터**: [데이터 담당자]
- **모델링**: [모델링 담당자]
- **인프라**: [인프라 담당자]

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](../../../LICENSE) 파일을 참조하세요.

## 🙏 기여하기

1. 이 저장소를 포크합니다
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

---

**📅 마지막 업데이트**: 2024년 12월 27일  
**🔄 버전**: 1.0.0  
**👥 기여자**: [팀원 명단] 
