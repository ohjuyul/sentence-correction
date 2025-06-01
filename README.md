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

`sentence/train` 및 `sentence/valid` 폴더에 JSON 파일을 준비한다.  

각 JSON 파일은 다음과 같은 형식을 가진다:

```json
{
  "ko": "오답 문장",
  "corrected": "교정된 문장"
}
```

예:

```json
{
  "ko": "나는 학교에 갔습니다.",
  "corrected": "나는 학교에 갔어요."
}
```

---

## 환경 설정

`requirements.txt` 내용:

```
absl-py==2.2.2
aiohappyeyeballs==2.6.1
aiohttp==3.11.18
aiosignal==1.3.2
attrs==25.3.0
certifi==2025.4.26
charset-normalizer==3.4.2
click==8.2.0
colorama==0.4.6
datasets==3.6.0
dill==0.3.8
et_xmlfile==2.0.0
filelock==3.18.0
frozenlist==1.6.0
fsspec==2025.3.0
huggingface-hub==0.31.2
idna==3.10
Jinja2==3.1.6
joblib==1.5.0
lxml==5.4.0
MarkupSafe==3.0.2
mpmath==1.3.0
multidict==6.4.3
multiprocess==0.70.16
networkx==3.4.2
nltk==3.9.1
numpy==2.2.5
nvidia-cublas-cu12==12.6.4.1
nvidia-cuda-cupti-cu12==12.6.80
nvidia-cuda-nvrtc-cu12==12.6.77
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.5.1.17
nvidia-cufft-cu12==11.3.0.4
nvidia-cufile-cu12==1.11.1.6
nvidia-curand-cu12==10.3.7.77
nvidia-cusolver-cu12==11.7.1.2
nvidia-cusparse-cu12==12.5.4.2
nvidia-cusparselt-cu12==0.6.3
nvidia-nccl-cu12==2.26.2
nvidia-nvjitlink-cu12==12.6.85
nvidia-nvtx-cu12==12.6.77
openpyxl==3.1.5
packaging==25.0
pandas==2.2.3
portalocker==3.1.1
propcache==0.3.1
pyarrow==20.0.0
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
rouge-score==0.1.2
sacrebleu==2.5.1
safetensors==0.5.3
scikit-learn==1.6.1
scipy==1.15.3
setuptools==68.1.2
six==1.17.0
sympy==1.14.0
tabulate==0.9.0
threadpoolctl==3.6.0
tokenizers==0.21.1
torch==2.7.0
tqdm==4.67.1
transformers==4.51.3
triton==3.3.0
typing_extensions==4.13.2
tzdata==2025.2
urllib3==2.4.0
wheel==0.42.0
xxhash==3.5.0
yarl==1.20.0

```

설치:

```bash
pip install -r requirements.txt
```

---

## 사용법

학습 실행:

```bash
python train.py
```

- 배치 크기: 16  
- 에폭 수: 9  
- 3 에폭마다 `checkpoints/` 폴더에 모델 저장  
- 학습 과정에서 손실, 정확도, BLEU 점수 출력  

---

## 주요 함수 및 클래스 설명

### model.py

- `CorrectionModel`  
  - BERT 인코더와 BART 디코더 포함  
  - `forward` 메서드에서 인코더 입력을 처리하고, 디코더 입력으로부터 예측 로그잇 반환  

- `BartDecoder`, `BartDecoderLayer`, `BartAttention`  
  - Transformer 디코더 구조 구현  
  - 멀티헤드 어텐션, 피드포워드 네트워크, 레이어 정규화, 드롭아웃 포함  

### data.py

- `GrammarCorrectionDataset`  
  - JSON 파일에서 문장 쌍 로드  
  - 토크나이저로 토큰화 및 패딩 후 텐서 반환  

### train.py

- `shift_tokens_right`  
  - 디코더 입력 생성용, 라벨 토큰 시퀀스 오른쪽 한 칸 이동 및 시작 토큰 삽입  

- `calculate_accuracy`  
  - 예측과 라벨의 일치도 마스킹 후 계산  

- `decode_tokens`  
  - 토큰 ID 시퀀스를 텍스트로 디코딩  

- `train` 함수  
  - 학습 및 검증 루프 구현  
  - tqdm 진행 표시  
  - BLEU 점수 계산 및 체크포인트 저장  

---

## 참고

- 데이터는 UTF-8 인코딩 권장  
- 모델은 한국어 BERT 기반  
- GPU 사용 가능 시 자동 활용  
- BLEU 점수는 smoothing function 적용하여 안정적 계산  

