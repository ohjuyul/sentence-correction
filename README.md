# Korean Grammar Correction Model

한국어 문법 오류 자동 교정을 위한 딥러닝 모델입니다.  
BERT 인코더와 BART 스타일 디코더를 결합하여 문장 내 오타 및 문법 오류를 효과적으로 수정합니다.

---

## 프로젝트 구성

- `model.py`  
  - **CorrectionModel** 클래스 포함  
  - BERT 기반 인코더(`beomi/kcbert-base`)와 BART 스타일 디코더 구현  
  - 디코더는 여러 개의 BartDecoderLayer로 구성되어 있으며, 각 레이어는 셀프 어텐션과 인코더-디코더 크로스 어텐션을 포함  
  - 출력층은 디코더 마지막 hidden state에 대해 linear projection으로 토큰별 logits 생성

- `data.py`  
  - **GrammarCorrectionDataset** 클래스 포함  
  - 지정한 폴더 내 JSON 파일들을 모두 로드하여 (오답 문장, 교정 문장) 쌍으로 관리  
  - 토크나이저(`beomi/kcbert-base`)를 사용해 입력과 정답 문장 토큰화 및 텐서 변환  
  - CrossEntropyLoss에 맞게 padding 토큰을 -100으로 마스킹 처리

- `train.py`  
  - 학습 루프와 검증 루프 포함  
  - 옵티마이저: AdamW, 학습률 스케줄러: Linear decay  
  - 배치 단위 학습 진행, 9 epoch 기본 설정  
  - 검증 시 BLEU 점수 및 정확도 계산하여 학습 성능 평가  
  - 3 epoch마다 모델 체크포인트 저장

---

## 데이터 준비

`sentence/train` 및 `sentence/valid` 폴더에 JSON 파일을 준비합니다.  

각 JSON 파일은 다음과 같은 형식을 가집니다:

```json
{
  "ko": "오답 문장",
  "corrected": "교정된 문장"
}
```

예를 들어:

```json
{
  "ko": "나는 학교에 갔습니다.",
  "corrected": "나는 학교에 갔어요."
}
```

---

## 환경 설정

`requirements.txt` 파일에 다음과 같이 작성하세요:

```
torch
transformers
tqdm
nltk
```

설치 명령:

```bash
pip install -r requirements.txt
```

---

## 사용법

학습은 다음 명령어로 실행합니다:

```bash
python train.py
```

- 기본 배치 크기: 16  
- 기본 에폭 수: 9  
- 학습 중간중간 `checkpoints/` 폴더에 모델이 저장됩니다 (3 epoch마다 저장)  
- 학습 과정에서 손실(loss), 정확도(accuracy), BLEU 점수가 출력됩니다  

---

## 주요 함수 및 클래스 설명

### model.py

- `CorrectionModel`  
  - BERT 인코더와 BART 디코더를 포함하는 모델  
  - `forward` 메서드에서는 인코더에 입력 문장을 넣고, 디코더에 디코더 입력 토큰을 넣어 예측 로그잇을 반환  

- `BartDecoder`, `BartDecoderLayer`, `BartAttention`  
  - Transformer 디코더 구조를 세부적으로 구현한 클래스들  
  - 멀티헤드 어텐션, 피드포워드 네트워크, 레이어 정규화, 드롭아웃 등을 포함  

### data.py

- `GrammarCorrectionDataset`  
  - 폴더 내 JSON 파일을 읽고, (오답 문장, 교정 문장) 쌍을 저장  
  - `__getitem__` 메서드에서 토크나이저를 사용해 문장을 토큰화 후 패딩하여 모델 입력에 맞게 반환  

### train.py

- `shift_tokens_right`  
  - 디코더 입력을 생성하기 위해, 라벨 토큰 시퀀스를 오른쪽으로 한 칸 이동시키고 시작 토큰을 채움  

- `calculate_accuracy`  
  - 예측과 실제 라벨 간 일치도를 마스킹하여 계산  

- `decode_tokens`  
  - 토크나이저를 통해 토큰 ID 시퀀스를 다시 텍스트로 변환  

- `train` 함수  
  - 학습 및 검증 루프 포함  
  - tqdm으로 진행 상황 표시  
  - 검증 시 BLEU 점수 계산하여 성능 확인  
  - 주기적으로 체크포인트 저장  

---

## 참고 사항

- 데이터 파일은 UTF-8 인코딩을 권장합니다.  
- 모델은 한국어 BERT 기반이므로 한국어 데이터에 최적화되어 있습니다.  
- 환경에 따라 GPU 사용 가능 시 자동으로 활용합니다.  
- BLEU 점수는 문장 단위로 계산하며, smoothing function을 사용해 안정성을 높였습니다.  

---

필요한 다른 도움이나 수정 사항 있으면 알려주세요!  
