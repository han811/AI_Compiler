### Problems
1. 

### Installation
```bash
docker run --gpus all -it --name ai_compiler -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /home/han811/Desktop/ws/AI_Compiler/:/workspace/AI_Compiler/ nvcr.io/nvidia/pytorch:24.02-py3
git clone https://github.com/han811/AI_Compiler.git
cd AI_Compiler
```

NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@b8eea8a


