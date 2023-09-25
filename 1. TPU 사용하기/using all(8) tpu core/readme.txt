
!wget https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl

!pip install torch_xla-2.0-cp310-cp310-linux_x86_64.whl

위 두줄 코드를 실행한다. (pip 설치 관련 오류가 발생해도 문제없이 import 가능)

해당 prediction.py,train.py와 modules디렉토리 내의 trainer.py, recorder.py 파일을 교체한다. 
config 폴더 내의 train_config.yaml파일을 교체한다. (shuffle option이 반드시 false상태여야 한다.)
dataset.py는 변경점은 없으나, augmentation코드에 함께 필요해서 넣어놨다. 사용 모델 또한 적절히 추가하는게 필요하다. 
여기서 train_config.yaml 파일은 ConvNext_v2기준으로 되어 있다.

