Our first hackathon project - Artificial Sensory Assistance (ASA)
------------------------------------
Utilizing the cutting-edge YOLOv8 technology combined with the powerful pyttsx3 library, our product detects objects and safeguards blind people about potential obstacles in their daily lives.

(Our model is stored in "model" folder)

Installing necessary libraries
```
pip install ultralytics pyttsx3
```
Allowing system to use GPU to process instead of CPU (if it is available on your computer)
(CUDA 11.8)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Access the location of the model
```
cd ./model
```
Run the model
```
python main.py
```
