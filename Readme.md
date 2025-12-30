# DerainSys v2.0 - 智能图像修复与处理系统

DerainSys 是一个集成了深度学习去雨、图像拼接、智能裁剪和色彩调整功能的一站式图像处理平台。系统采用前后端分离架构，前端基于 **Vue 3 + Element Plus** 构建了现代化的 Glassmorphism (磨砂玻璃) 风格界面，后端基于 **Spring Boot** 提供 robust 的服务支持，AI 引擎采用改进的 **Attention Enhanced UNet** 架构。

![DerainSys Demo](https://via.placeholder.com/800x400?text=DerainSys+v2.0+UI+Preview)

## ✨ 主要功能 (Key Features)

### 1. 🌧️ 智能去雨 (Smart Deraining)
*   **核心算法**：集成 CBAM (Convolutional Block Attention Module) 的增强型 UNet，能精准识别并去除雨纹。
*   **无损输出**：采用 **"Safe Margin" (安全边距)** 填充策略，在推理前自动扩展边缘并在推理后精准裁剪，彻底消除 AI 模型常见的边缘伪影 (Purple Artificats)，保持原图分辨率。

### 2. 🧩 图片拼接 (Image Stitching)
*   **多图整合**：支持批量上传多张图片。
*   **灵活排序**：支持拖拽或点击按钮调整图片顺序。
*   **双模式拼接**：
    *   **竖向拼接**：适合长图制作、聊天记录合并。
    *   **横向拼接**：适合全景展示、对比图制作。

### 3. ✂️ 图片裁剪 (Image Cropping)
*   **双模式裁剪**：
    *   **矩形裁剪**：标准比例或自由比例矩形框选。
    *   **自由形状 (Polygon)**：通过 Canvas 绘制任意多边形路径进行不规则抠图，支持自定义锚点。

### 4. 🎨 色彩调整 (Color Adjustment)
*   **实时滤镜**：提供亮度 (Brightness)、对比度 (Contrast)、饱和度 (Saturation) 的精细调节。
*   **黑白模式**：一键转换为灰度图。
*   **高清导出**：所有调整均基于 Canvas 进行像素级处理，确保导出画质。

---

## 📂 项目结构 (Project Structure)

```
derainSys/
├── ai_engine/          # AI 核心模块
│   ├── Derain_model.py # Attention Enhanced UNet 模型定义
│   ├── train.py        # 训练脚本
│   └── dataset/        # 训练/测试数据集
├── backend/            # Java Spring Boot 后端
│   └── src/main/java/com/example/demo/controller/ImageRestoreController.java # API 接口
├── frontend/           # Vue 3 前端
│   ├── src/views/
│   │   ├── ImageCrop.vue    # 图片裁剪页面
│   │   ├── ImageStitch.vue  # 图片拼接页面
│   │   └── ImageAdjust.vue  # 色彩调整页面
│   └── src/components/
│       └── ImageRestore.vue # 智能去雨页面
└── Readme.md           # 项目文档
```

---

## 🛠️ 环境要求 (Prerequisites)

### AI 引擎 (Python)
*   **Python**: 3.8+
*   **PyTorch**: 2.0.0+ (建议搭配 CUDA 11.8)
*   **依赖库**: `torchvision`, `Pillow`, `tqdm`
    *   *注：v2.0 已移除 Mamba 架构依赖，无需安装复杂的编译环境，纯 PyTorch 即可运行。*

### 后端 (Java)
*   **JDK**: 1.8+
*   **Maven**: 3.x

### 前端 (Node.js)
*   **Node.js**: 16+ (LTS)
*   **npm**: 包管理器

---

## 🚀 快速开始 (Getting Started)

### 1. 启动后端 (Backend)
```bash
cd backend
mvn spring-boot:run
```
服务默认运行在 `http://localhost:8080`。

### 2. 启动前端 (Frontend)
```bash
cd frontend
npm install
npm run serve
```
访问终端输出的地址 (通常为 `http://localhost:8081`) 即可使用系统。

---

## 💡 开发日志 (Changelog)

### v2.1 (2025-12-30) - UI Polish & Bug Fixes
*   [Update] 全面优化各页面布局，使内容更加饱满、居中美观。
*   [Update] 为所有按钮增加 **动态效果**：shimmer 光效、hover 放大、icon 跳动动画。
*   [Update] 图片拼接页面：居中方向选择器，增加底部分隔线装饰。
*   [Update] 图片裁剪页面：修复实时预览与选区不一致的 Bug，改用 `getCropData` 获取真实裁剪预览。
*   [Update] 色彩调整页面：修复路由路径问题 (`/color` → `/adjust`)，页面正常显示。
*   [Update] 智能去雨页面：限制图片预览大小防止溢出，固定按钮区域高度防止布局抖动。
*   [Fix] 修复侧边栏导航图标缺失问题（图片裁剪使用 `Crop` 图标，色彩调整使用 `Setting` 图标）。
*   [Fix] 移除各上传区域默认白色背景板，统一透明磨砂玻璃风格。
*   [Style] 优化 Tailwind CSS 配置，兼容 v4 版本的插件与导入语法。

### v2.0 (2025-12-28)
*   [New] 新增 **图片拼接** 功能模块。
*   [Update] 全面升级 UI 为 **Glassmorphism (磨砂玻璃)** 风格，增加动态背景。
*   [Update] **图片裁剪** 增加多边形 (Polygon) 自由裁剪模式。
*   [Fix] 修复 AI 模型边缘紫色伪影问题 (Safe Margin 实现)。
*   [Refactor] 移除 Mamba 依赖，迁移至更通用的 Attention UNet。
*   [Docs] 更新项目文档。

---

## © Copyright
**Powered by DerainSys © 2025 | 合肥工业大学梁禹设计**