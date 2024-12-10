# Korean Sign-Language-Translator

<br>
<hr>

## 설계 과정 

- AI HUB의 수어 영상 데이터 셋

- 수어 단어 데이터의 부족으로 수어 문장에서 적어도 300번 이상 발생한 단어를 뽑아서 훈련 데이터로 사용 

- MediaPipe의 Hollistic 모델로 컴퓨터 웹캠으로 부터 상체 POSE, FACE, LEFT HAND RIGHT HAND 키포인트 추출

- ~50개의 수어 단어 번역 기능

- 모델은 Google colab에서 훈련시켰고, 그 모델을 TFLITE model으로 빼서 구현을 진행함 

- Flask-SocketIO으로 간단한 웹페이지 구현

<br>
<hr>

## 동작과정

- 시작하기 버튼을 통해서 Client의 Webcam 에 접근 권한을 얻음.
- 모델을 로드 시킨 뒤, 손이 카메라에 잡히면 recording 시작
- 손이 카메라에 안찍히면 그 때까지의 sequence를 predict해서 특정 threshold를 넘으면 문장에 추가함

<br>
<hr>

## TODO
- Gloss translation using ChatGPT
- Training more accurate models


