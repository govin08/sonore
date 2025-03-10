# 2025. 2. 12
관심있는 프로젝트 목록 정리
- 음성 명령어 인식 (Voice Command Recognition)
- 음악 장르 분류 (Music Genre Classification)
- 배경 소음 제거 (Background Noise Reduction)
- voice conversion / voice cloning
- audio resolution enhancement

# 2025. 3. 10
repository 생성 (sonore)

괜찮은 데이터셋 발견 및 탐색
- [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50)
- 2000개의 5초 길이의 소리 데이터. 각 소리는 50 종류의 라벨이 붙어있다.
    - dog, rooster, pig, cow, ...
    - rain, sea waves, crickets, ...
    - ...
- 따라서 이 데이터셋을 분류하는 문제를 생각해볼 수 있다.
- 이 데이터셋을 만든 것으로 추정되는 사람은 Karol Piczak으로 바르샤바 대학교 소속이다. 이 사람은 이 데이터셋으로 [ESC: Dataset for Environmental Sound Classification](https://www.karolpiczak.com/papers/Piczak2015-ESC-Dataset.pdf)라는 논문을 냈다. 이 논문은 2015년에 ACM에서 발표된 논문인 것으로 보인다.
- 10년 전 데이터셋이지만 해당 논문의 피인용횟수가 1895회인 것으로 보아 상당히 중요한 데이터셋인 것으로 보인다. 따라서 이 데이터셋을 가지고 여러가지를 해보는 것이 의미있을 것으로 보인다.

ESC-50 탐색
- 데이터 다운로드
- gitignore 설정정
- ./audio의 2000개 소리 중 dog 범주에 속하는 소리 청취