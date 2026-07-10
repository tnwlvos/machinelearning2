# Machine Learning 2

인공지능설계실습2 과목에서 수행한 딥러닝 실습 및 텀프로젝트 저장소입니다.  
RNN, Transformer, AutoEncoder, GAN, CNN, YOLO 등 다양한 딥러닝 모델을 실습하고, Term Project로 Transformer 기반 주가 예측 모델을 구현하였습니다.

## Project Overview

본 저장소는 딥러닝 모델의 구조를 이해하고, 실제 데이터에 적용해보는 것을 목표로 구성되었습니다.  
특히 Term Project에서는 Yahoo Finance API 기반 시계열 데이터를 수집하고, Transformer 모델을 활용하여 주가 예측을 수행했습니다.

## Main Contents

- RNN.ipynb
  - RNN 기반 시계열 모델 실습

- Transformer.ipynb
  - Transformer 구조 실습
  - Positional Encoding
  - Multi-Head Attention

- TermProject.ipynb
  - 주가 예측 텀프로젝트 초기 버전

- TermProject_ver2.ipynb
  - 주가 예측 텀프로젝트 개선 버전

- AE.ipynb
  - AutoEncoder 실습

- GAN.ipynb
  - GAN 구조 실습

- chap5_R_CNN.ipynb
  - R-CNN 관련 실습

- char4_cnn.ipynb
  - CNN 실습

- yolo.ipynb
  - YOLO 기반 객체 탐지 실습

## Term Project: Transformer Stock Prediction

Transformer 기반 시계열 예측 모델을 구현하여 주가 예측을 수행했습니다.

### Main Process

1. Yahoo Finance API 기반 주가 데이터 수집
2. 데이터 전처리 및 정규화
3. Sliding Window 기반 입력 시퀀스 구성
4. Transformer Encoder 구조 적용
5. Multi-Head Attention 및 Positional Encoding 적용
6. Hyperparameter 조정
7. 예측 결과 분석

## Tech Stack

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- PyTorch
- TensorFlow
- Yahoo Finance API
- Transformer
- RNN
- CNN
- GAN
- AutoEncoder
- YOLO

## Key Implementation Points

- 시계열 데이터 전처리
- Sliding Window 기반 입력 구조 설계
- Transformer 기반 예측 모델 구현
- Attention 구조 이해 및 적용
- 하이퍼파라미터 조정을 통한 성능 개선
- 다양한 딥러닝 모델 구조 비교 학습

## Result

본 프로젝트를 통해 단순 모델 사용을 넘어, 데이터 수집부터 전처리, 모델 구성, 학습, 예측 결과 분석까지 이어지는 AI 모델 개발 흐름을 경험하였습니다.  
특히 Transformer 구조를 시계열 예측 문제에 적용하며 Attention 기반 모델의 특징을 학습했습니다.
