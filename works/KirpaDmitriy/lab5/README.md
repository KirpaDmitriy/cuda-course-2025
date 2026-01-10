### torch -> onnx -> engine -> .cu

1) Скачиваем картинки из Coco
2) Калибруемся на них под fp8
3) Получаем engine
4) В ядре в .cu вызываем сетку на каждый кадр

Видео в папке ./videos.

Запуск:

```
pip install torch torchvision numpy opencv-python pycuda requests
pip install onnx onnxscript
pip install nvidia-pyindex 
pip install nvidia-tensorrt 


cd scripts
python3 0_download_data.py
python3 1_export_onnx.py
python3 2_build_int8_engine.py
cd ..

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt update
sudo apt install libopencv-dev

cd build
cmake ..
make -j$(nproc)

./RetinaNetTRT ../models/retinanet_int8.engine ../models/classes.txt ../test.mp4 ../result.mp4
```