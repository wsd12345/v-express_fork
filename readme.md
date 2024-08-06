# 一、环境安装
- python==3.12.2

1. 基础库已经在`requirements.txt`中提供，可以通过如下命令安装
    ```bash
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
2. 其中dlib库安装可能出现问题，提供了whl文件，使用下面命令安装，不同系统的分隔符可能不同
    ```bash
    pip install third_part\whl\dlib-19.24.99-cp312-cp312-win_amd64.whl
    ```
3. 其中onnxruntime库安装可能出现问题，提供了gpu版本的whl文件，使用下面命令安装，不同系统的分隔符可能不同。
   注意：在python3.12.2中onnxruntime叫做ort_nightly。
   ```bash
   pip install third_part\whl\ort_nightly_gpu-1.19.0.dev20240531001-cp312-cp312-win_amd64.whl
   ```
4. 安装需要的深度学习框架，这里用的是`torch==2.3.1+cu121`，可以使用以下命令安装
    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    ```

5. 安装ffmpeg并添加到环境变量。可以访问[ffmpeg官网](https://ffmpeg.org/)进行安装。
6. 如果需要使用其他框架可以自行安装。如果发现安装失败，可以更换镜像源，修改`-i`后的参数，现在用的是清华源。
# 二、快速开始
1. 给定一段wav格式的音频或者带有mp4格式的视频作为音频素材。
2. 给定一个图片或者一个视频片段，需要可以检测到人脸，其视频别太长，太长会爆内存，还没做优化。 


# 三、相关工作
请参考[三方库](./third_part)中的工作。本项目是修改于[V-Express](./third_part/V_Express)。
V-Express的工作使用扩散模型生产面部图像，[原项目地址](https://github.com/tencent-ailab/V-Express.git)。
原项目使用kps进行眨眼指导，使用音频进行嘴部动作的指导，使用一张参考图像指导进行去噪采样。
本项目的改进：

1. 使用一段视频进行指导去噪采样。
2. 此外发现，使用kps可以生成与参考视频头部动作更加接近的视频。不使用kps时，若参考视频头部运动幅度过大，生成视频则不会有同样的运动幅度。
3. 对于流式输出，因为需要计算上下文信息，所以没办法做到流式输出。当`CONTEXT_OVERLAP=0`时，不使用上下文信息，则会生成跳变的视频。
4. 发现生成的脸部具有比参考视频更亮。不知道是不是参考视频的原因。



# 四、建议使用
1. 对于扩散模型一次处理12张$512 \times 512$的面部图像，需要显存12.8g。
2. 关于kps指导头部动作，没有经过大规模测试。
3. 如果想要更换背景，请使用绿布背景的指导视频。
4. 原项目说可以是试试$384 \times 384$的以加快去噪速度，实测下来，生成效果不好。如果想要修改图像大小，请参考[magic-animate](https://github.com/magic-research/magic-animate)。
