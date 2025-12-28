<template>
  <div class="page-container">
     <div class="glass-card main-card">
        <h1 class="page-title"><el-icon style="vertical-align: middle; margin-right: 8px"><Operation /></el-icon> 色彩调整 (Color Adjustment)</h1>
        
        <!-- Upload Area -->
        <div v-if="!currentUrl" class="upload-area">
             <el-upload
                class="upload-demo"
                drag
                action=""
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileChange"
            >
                <div class="upload-content animated-upload">
                    <el-icon class="upload-icon"><UploadFilled /></el-icon>
                    <div class="upload-text">点击或拖拽上传图片</div>
                    <div class="upload-subtext">支持 JPG/PNG 高清原图</div>
                </div>
            </el-upload>
        </div>

        <div v-else class="workspace">
            <!-- Canvas/Image Preview -->
             <div class="image-wrapper">
                 <img ref="previewImg" :src="currentUrl" :style="filterStyle" />
             </div>

             <!-- Controls -->
             <div class="controls-panel">
                 <div class="control-group">
                     <span class="label">黑白模式 (Grayscale)</span>
                     <el-switch v-model="settings.grayscale" active-color="#13ce66" />
                 </div>
                 
                 <div class="control-group">
                     <span class="label">亮度 (Brightness) - {{ settings.brightness }}%</span>
                     <el-slider v-model="settings.brightness" :min="0" :max="200" />
                 </div>

                 <div class="control-group">
                     <span class="label">对比度 (Contrast) - {{ settings.contrast }}%</span>
                     <el-slider v-model="settings.contrast" :min="0" :max="200" />
                 </div>
                 
                 <div class="control-group">
                     <span class="label">饱和度 (Saturation) - {{ settings.saturate }}%</span>
                     <el-slider v-model="settings.saturate" :min="0" :max="200" />
                 </div>

                 <div class="action-buttons">
                     <el-button class="action-btn primary-btn" type="primary" @click="downloadResult">
                        <el-icon style="margin-right: 8px"><Download /></el-icon> 下载结果
                     </el-button>
                     <el-button class="reset-btn" type="danger" plain @click="reset">
                        <el-icon style="margin-right: 8px"><Delete /></el-icon> 重置
                     </el-button>
                 </div>
             </div>
        </div>
     </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { UploadFilled, Download, Delete, Operation } from '@element-plus/icons-vue'


const currentUrl = ref('')
const file = ref(null)

const settings = ref({
    grayscale: false,
    brightness: 100,
    contrast: 100,
    saturate: 100
})

const filterStyle = computed(() => {
    const filters = [
        `brightness(${settings.value.brightness}%)`,
        `contrast(${settings.value.contrast}%)`,
        `saturate(${settings.value.saturate}%)`,
        settings.value.grayscale ? 'grayscale(100%)' : 'grayscale(0%)'
    ]
    return {
        filter: filters.join(' ')
    }
})

function handleFileChange(uploadFile) {
    if(!uploadFile.raw.type.startsWith('image/')) return
    file.value = uploadFile.raw
    currentUrl.value = URL.createObjectURL(uploadFile.raw)
}

function reset() {
    file.value = null
    currentUrl.value = ''
    settings.value = { grayscale: false, brightness: 100, contrast: 100, saturate: 100 }
}

function downloadResult() {
    const img = new Image()
    img.src = currentUrl.value
    img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        canvas.width = img.width
        canvas.height = img.height
        
        ctx.filter = filterStyle.value.filter
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'adjusted_image.png'
            a.click()
        })
    }
}
</script>

<style scoped>
.page-container {
    display: flex;
    justify-content: center;
    padding: 40px;
    width: 100%;
    min-height: 80vh;
    box-sizing: border-box;
}

.main-card {
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    animation: fadeIn 0.6s ease-out;
    display: flex;
    flex-direction: column;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.page-title {
    text-align: center;
    margin-bottom: 24px;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

/* Upload Area */
.upload-area {
    display: flex;
    justify-content: center;
    padding: 60px 0;
}
.upload-demo :deep(.el-upload-dragger) {
    background: rgba(255, 255, 255, 0.03);
    border: 2px dashed rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    height: 240px;
    width: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
}
.upload-demo :deep(.el-upload-dragger:hover) {
    border-color: #a5b4fc;
    background: rgba(165, 180, 252, 0.1);
    transform: translateY(-4px);
}
.upload-content {
    text-align: center;
    color: #e0e7ff;
}
.upload-icon {
    font-size: 56px;
    margin-bottom: 16px;
    color: #818cf8;
}
.upload-text {
    font-size: 18px;
    font-weight: 500;
}
.upload-subtext {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    margin-top: 8px;
}

.workspace {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 30px;
    /* height: 600px; Remove fixed height to adapt content */
}

.image-wrapper {
    background: #1a1a1a;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 400px;
}
.image-wrapper img {
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
}

/* Controls Side */
.controls-panel {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    border: 1px solid rgba(255,255,255,0.05);
}

.control-group {
    /* margin-bottom: 20px; */
}
.label {
    display: block;
    margin-bottom: 8px;
    font-size: 14px;
    color: #e0e7ff;
    font-weight: 500;
}

.action-buttons {
    margin-top: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.action-btn {
    width: 100%;
    height: 48px;
    font-size: 16px;
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}

.primary-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}
.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.6);
}

.reset-btn {
    width: 100%;
}
</style>
