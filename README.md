## car-damage-segmentation

- **Damage_segmentation_with_Unet_final_results_only.ipynb** : 진행했던 코드 작업에 대한 단계적 설명이 있는 파일입니다. 작업 내용에 대해 살펴보시려면 본 파일을 읽어주세요. _(수정) 기존 파일에서 최종 학습 결과에 대한 logging만 남겨두었습니다._

- 디렉토리 구조 및 파일별 역할

|-- code

|　　|-- src

|　　|　　|--```Datasets.py``` : Custom dataset 정의, 데이터셋 resize, transform

|　　|　　|--```Evaluation.py``` : 모델 성능 검증 코드

|　　|　　|--```Models.py``` : smp 라이브러리를 활용한 Unet 모델 클래스

|　　|　　|--```Models_implementation.py``` : 논문 기반으로 구현한 Unet 모델 클래스 (w. batchnorm)

|　　|　　|--```Train.py``` : train loop, dataloader, validation 

|　　|　　|--```Utils.py``` : Annotation 형식 변환, loss & metric 계산 함수들

|　　|-- ```main.py```

|　　|-- ```damage_labeling.csv``` : raw annotation 파일 -> 추후 가공을 통해 ```data/datainfo```의 json 파일들로 변경됨

|-- data / Dataset

　　　　　|-- 1.원천데이터 : 원본 이미지 (.jpg)

　　　　　|-- 2.라벨링데이터 : 어노테이션 (.json)
