# Image Deraining System

一个基于 **Mamba (State Space Model)** 和 **UNet** 架构的深度学习图像去雨系统，包含完整的 Vue 3 前端界面和 Spring Boot 后端服务。

## 📂 项目结构 (Project Structure)

经过优化，本项目分为三个主要模块：

*   **`ai_engine/` (AI 核心)**
    *   包含所有深度学习相关代码 (`mamba_model.py`, `train.py`, `test.py`)。
    *   存放数据集 (`dataset/`) 和模型权重 (`*.pth`)。
    *   核心算法：结合 Haar 小波变换和 Vision Mamba 模块的 UNet 架构。

*   **`backend/` (后端服务)**
    *   以前的 `qianhouduan` 目录。
    *   基于 **Java Spring Boot**。
    *   负责提供 API 接口，并调用 Python 脚本 (`your_model_script.py`) 执行推理任务。

*   **`frontend/` (前端界面)**
    *   基于 **Vue.js 3** + **Element Plus**。
    *   用户可以通过 Web 界面上传图片并实时查看去雨效果。

---

## 🛠️ 环境要求 (Prerequisites)

为了运行整个系统，您需要配置以下环境：

### AI 引擎 (Python)
*   **Python**: 3.8
*   **CUDA**: 11.8 (推荐)
*   **PyTorch**: 2.0.0
*   **核心依赖**:
    *   `mamba_ssm`
    *   `causal_conv1d`
    *   `torchvision`
    *   `Pillow`, `tqdm` 等

### 后端 (Java)
*   **JDK**: 1.8 或更高版本
*   **Maven**: 用于构建项目

### 前端 (Node.js)
*   **Node.js**: 建议使用 LTS 版本
*   **npm**: 包管理器

---

## 🚀 快速开始 (Getting Started)

### 1. 准备 AI 环境
请确保您的 Python 环境安装了正确的依赖。
```bash
# 进入 AI 引擎目录
cd ai_engine

# (可选) 建议使用 Conda 创建环境
conda create -n derain python=3.8
conda activate derain

# 安装 PyTorch (示例，请根据您的 CUDA 版本调整)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装 Mamba 相关库 (请参考 run_config.sh 中的 whl 文件或官方文档进行安装)
pip install packaging einops
# 注意：Mamba 在 Windows 下安装可能比较繁琐，建议使用预编译的 whl 包
```

### 2. 启动后端
```bash
cd backend
mvn spring-boot:run
```
后端启动后默认监听 `8080` 端口。

### 3. 启动前端
```bash
cd frontend
# 安装依赖
npm install
# 启动开发服务器
npm run serve
```
前端启动后通常访问 `http://localhost:8080` (如果是 8080 被后端占用，可能会自动切换到 8081)。

---

## 📊 模型训练与测试 (Training & Testing)

如果您想自己训练模型或运行测试脚本：

### 训练 (Train)
将训练数据放入 `ai_engine/dataset/train` 目录 (需包含 `rain` 和 `gt` 子目录)。
```bash
cd ai_engine
python train.py
```

### 测试 (Test)
将测试数据放入 `ai_engine/dataset/test` 目录。
```bash
cd ai_engine
python test.py
```

---

## 📝 原始环境备注
> 以下是项目初始记录的特定环境版本，供参考：
*   Pytorch 2.0.0
*   Python3.8
*   Cuda 11.8
*   `causal_conv1d-1.1.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64`
*   `mamba_ssm-1.1.3+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64`