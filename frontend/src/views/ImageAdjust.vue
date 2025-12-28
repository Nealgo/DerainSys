<template>
  <div class="page-container">
     <div class="glass-card main-card">
        <h1 class="page-title">色彩调整 (Color Adjustment)</h1>
        
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
                <div class="upload-content">
                    <i class="el-icon-upload upload-icon"></i>
                    <div class="upload-text">点击上传图片</div>
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
                     <el-button type="danger" plain @click="reset">重置</el-button>
                     <el-button type="primary" @click="downloadResult">下载结果</el-button>
                 </div>
             </div>
        </div>
     </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'


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
    padding-top: 20px;
    width: 100%;
}
.main-card {
    width: 100%;
    max-width: 900px;
    padding: 30px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    animation: slideUp 0.6s cubic-bezier(0.25, 1, 0.5, 1) forwards;
    opacity: 0;
    transform: translateY(30px);
}
@keyframes slideUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.page-title {
    text-align: center;
    margin-bottom: 30px;
    font-size: 24px;
    font-weight: 600;
}
.upload-area {
    display: flex;
    justify-content: center;
}
.upload-demo :deep(.el-upload-dragger) {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.3);
}
.upload-icon { font-size: 40px; margin-bottom: 10px; }

.workspace {
    display: flex;
    gap: 30px;
}
.image-wrapper {
    flex: 2;
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    padding: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}
.image-wrapper img {
    max-width: 100%;
    max-height: 500px;
    object-fit: contain;
}
.controls-panel {
    flex: 1;
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
}
.control-group {
    margin-bottom: 20px;
}
.label {
    display: block;
    margin-bottom: 8px;
    font-size: 14px;
    color: #ddd;
}
.action-buttons {
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
}
</style>
