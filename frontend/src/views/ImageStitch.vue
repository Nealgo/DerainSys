<template>
  <div class="page-container">
    <div class="glass-card main-card">
      <h1 class="page-title"><el-icon style="vertical-align: middle; margin-right: 8px"><Picture /></el-icon> 图片拼接 (Image Stitching)</h1>
      
      <div class="stitch-layout">
        <!-- Left Panel: Upload & List -->
        <div class="panel upload-panel">
            <el-upload
                class="upload-demo"
                action=""
                multiple
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileChange"
            >
               <el-button class="action-btn primary-btn" type="primary">
                   <el-icon style="margin-right: 6px"><Plus /></el-icon> 添加图片
               </el-button>
            </el-upload>
            
            <div class="image-list" v-if="imageList.length > 0">
                <div v-for="(img, index) in imageList" :key="img.id" class="image-item">
                    <div class="thumbnail">
                        <img :src="img.url" />
                    </div>
                    <div class="item-info">
                        <span class="filename">{{ img.file.name }}</span>
                        <span class="filesize">{{ (img.file.size / 1024).toFixed(1) }} KB</span>
                    </div>
                    <div class="item-actions">
                        <el-button size="small" circle @click="moveUp(index)" :disabled="index === 0">
                            <el-icon><Top /></el-icon>
                        </el-button>
                        <el-button size="small" circle @click="moveDown(index)" :disabled="index === imageList.length - 1">
                            <el-icon><Bottom /></el-icon>
                        </el-button>
                        <el-button size="small" circle type="danger" plain @click="removeImage(index)">
                            <el-icon><Delete /></el-icon>
                        </el-button>
                    </div>
                </div>
            </div>
            <div v-else class="empty-list">
                <el-icon class="empty-icon"><Picture /></el-icon>
                <p>暂无图片，请点击上方按钮添加</p>
            </div>
        </div>

        <!-- Right Panel: Settings & Preview -->
        <div class="panel preview-panel">
            <div class="configuration">
                <span class="label">拼接方向：</span>
                <el-radio-group v-model="stitchMode" size="large">
                    <el-radio-button label="vertical">
                         <el-icon style="vertical-align:middle;margin-right:4px"><Bottom /></el-icon> 竖向拼接
                    </el-radio-button>
                    <el-radio-button label="horizontal">
                         <el-icon style="vertical-align:middle;margin-right:4px"><Right /></el-icon> 横向拼接
                    </el-radio-button>
                </el-radio-group>
            </div>

            <div class="preview-stage">
                 <div class="canvas-wrapper" v-if="previewUrl">
                     <img :src="previewUrl" class="stitch-result" />
                 </div>
                 <div v-else class="placeholder-preview">
                     <span>点击“开始拼接”生成预览</span>
                 </div>
            </div>

            <div class="action-footer">
                <el-button class="action-btn primary-btn" type="primary" :disabled="imageList.length < 2" @click="processStitch">
                    <el-icon style="margin-right: 8px"><MagicStick /></el-icon> 开始拼接
                </el-button>
                <el-button class="action-btn success-btn" type="success" :disabled="!previewUrl" @click="downloadResult">
                    <el-icon style="margin-right: 8px"><Download /></el-icon> 下载结果
                </el-button>
            </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { Plus, Delete, Top, Bottom, Picture, MagicStick, Download, Right } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const imageList = ref([])
const stitchMode = ref('vertical') // 'vertical' | 'horizontal'
const previewUrl = ref('')
const loading = ref(false)

let stitchedBlob = null

function handleFileChange(file) {
    if(!file.raw.type.startsWith('image/')) {
        ElMessage.warning('请选择图片文件')
        return
    }
    const url = URL.createObjectURL(file.raw)
    imageList.value.push({
        id: Date.now() + Math.random(),
        file: file.raw,
        url: url
    })
    // Reset preview on change
    previewUrl.value = ''
    stitchedBlob = null
}

function removeImage(index) {
    imageList.value.splice(index, 1)
    previewUrl.value = ''
}

function moveUp(index) {
    if(index <= 0) return
    const temp = imageList.value[index]
    imageList.value[index] = imageList.value[index-1]
    imageList.value[index-1] = temp
    previewUrl.value = ''
}

function moveDown(index) {
    if(index >= imageList.value.length - 1) return
    const temp = imageList.value[index]
    imageList.value[index] = imageList.value[index+1]
    imageList.value[index+1] = temp
    previewUrl.value = ''
}

async function processStitch() {
    if(imageList.value.length < 2) return
    loading.value = true
    
    try {
        // Load all images
        const loadedDigs = await Promise.all(imageList.value.map(item => loadImage(item.url)))
        
        // Calculate dimensions
        let totalWidth = 0
        let totalHeight = 0
        
        if(stitchMode.value === 'vertical') {
            // Width = Max Width, Height = Sum Height
            totalWidth = Math.max(...loadedDigs.map(img => img.width))
            totalHeight = loadedDigs.reduce((sum, img) => sum + img.height, 0)
        } else {
            // Width = Sum Width, Height = Max Height
            totalWidth = loadedDigs.reduce((sum, img) => sum + img.width, 0)
            totalHeight = Math.max(...loadedDigs.map(img => img.height))
        }
        
        const canvas = document.createElement('canvas')
        canvas.width = totalWidth
        canvas.height = totalHeight
        const ctx = canvas.getContext('2d')
        
        // Fill background (optional, maybe white or transparent)
        // ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,totalWidth, totalHeight);

        let currentX = 0
        let currentY = 0
        
        loadedDigs.forEach(img => {
            if(stitchMode.value === 'vertical') {
                // Draw centered horizontally? Or aligned left? Let's align left for now, or resize to fit?
                // Simple stitch: align left/top
                ctx.drawImage(img, 0, currentY)
                currentY += img.height
            } else {
                ctx.drawImage(img, currentX, 0)
                currentX += img.width
            }
        })
        
        canvas.toBlob(blob => {
            stitchedBlob = blob
            previewUrl.value = URL.createObjectURL(blob)
            loading.value = false
            ElMessage.success('拼接完成')
        }, 'image/png')
        
    } catch (e) {
        console.error(e)
        ElMessage.error('处理出错')
        loading.value = false
    }
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image()
        img.onload = () => resolve(img)
        img.onerror = reject
        img.src = src
    })
}

function downloadResult() {
    if(!stitchedBlob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(stitchedBlob)
    a.download = `stitch_${Date.now()}.png`
    a.click()
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

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
    display: flex;
    flex-direction: column;
    animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.page-title {
    text-align: center;
    margin-bottom: 30px;
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

.stitch-layout {
    display: grid;
    grid-template-columns: 1fr 2fr; /* 1:2 ratio for list vs preview */
    gap: 30px;
    min-height: 500px;
}

.panel {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    display: flex;
    flex-direction: column;
}

/* Left Panel */
.upload-panel {
    gap: 16px;
}
.image-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 500px;
}
.image-item {
    display: flex;
    align-items: center;
    background: rgba(0,0,0,0.2);
    padding: 8px;
    border-radius: 8px;
    gap: 10px;
}
.thumbnail {
    width: 48px;
    height: 48px;
    border-radius: 4px;
    overflow: hidden;
    flex-shrink: 0;
}
.thumbnail img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.item-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
.filename {
    font-size: 13px;
    color: #e0e7ff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.filesize {
    font-size: 11px;
    color: #9ca3af;
}
.item-actions {
    display: flex;
    gap: 4px;
}

.empty-list {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: rgba(255,255,255,0.3);
    border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 12px;
}
.empty-icon {
    font-size: 40px;
    margin-bottom: 10px;
}

/* Right Panel */
.preview-panel {
    gap: 20px;
}
.configuration {
    display: flex;
    align-items: center;
    gap: 12px;
}
.label {
    color: #e0e7ff;
    font-weight: 500;
}

.preview-stage {
    flex: 1;
    background: #1a1a1a;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    min-height: 300px;
}

.placeholder-preview {
    color: rgba(255,255,255,0.4);
    font-size: 14px;
}

.canvas-wrapper {
    max-width: 100%;
    max-height: 500px;
    overflow: auto;
    padding: 20px;
}
.stitch-result {
    max-width: 100%;
    display: block;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
}

.action-footer {
    display: flex;
    gap: 16px;
    justify-content: flex-end;
}

.action-btn {
    height: 44px;
    font-size: 15px;
    border-radius: 8px;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.primary-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    flex: 1;
}
.success-btn {
    flex: 1;
}
</style>
