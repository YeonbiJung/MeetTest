
!wget https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl

!pip install torch_xla-2.0-cp310-cp310-linux_x86_64.whl

위 두줄 코드를 실행한다. (pip 설치 관련 오류가 발생해도 문제없이 import 가능)

해당 prediction.py,train.py와 modules디렉토리 내의 trainer.py, recorder.py 파일을 교체한다. config 폴더 내의 train_config.yaml파일을 교체한다. (shuffle option이 반드시 false상태여야 한다.)

