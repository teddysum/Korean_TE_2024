# 함의 분석 결과 설명문 생성 Baseline
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '함의 분석 결과 설명문 생성'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습, 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.   

|Model|ROUGE-1|
|:---|---|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (without SFT)|0.15514|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (with SFT)|0.50358|
## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py

# 학습에 사용될 커스텀 함수들을 보관하는 디렉토리
src
├── data.py     # Custom Dataset
└── utils.py
```

## 데이터 (Data)
```
{
    "id": "nikluge-2024-함의분석결과설명문생성-train-000001",
    "input": {
        "premise": "또한 ‘브람스를 좋아하세요?’는 SBS 단편드라마 ‘17세의 조건’으로 감각적 연출을 선보인 조영민 감독과 섬세한 필력을 자랑한 류보리 작가가 의기투합해 웰메이드 작품 탄생을 예고하고 있다. 박은빈과 김민재의 신선한 조합에, SBS 기대주 감독과 작가가 가세한 SBS 새 월화드라마 ‘브람스를 좋아하세요?’는 8월 31일에 첫 방송된다.",
        "proposition": "박은빈이 출연한 SBS 월화 드라마의 제목에는 작곡가 이름이 들어간다.",
        "label": "entailment"
    },
    "output": "'브람스를 좋아하세요?'에 독일 작곡가인 브람스의 이름이 들어가 있다. 따라서 가설 문장은 함의에 해당한다."
}
```

## 실행 방법 (How to Run)
### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=1,3 python -m run.train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 5 \
    --lr 2e-5 \
    --warmup_steps 20
```

### 추론 (Inference)
```
python -m run.test \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```

## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
