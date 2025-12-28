<template>
  <div class="app-container">
    <div class="glass-card">
      <h1 class="title">AI Image Deraining</h1>
      <p class="subtitle">智能去雨 · 细节还原 · 极致体验</p>

      <!-- 上传区域 -->
      <div v-if="!originalUrl" class="upload-area">
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
            <div class="upload-text">点击或拖拽上传图片</div>
            <div class="upload-hint">支持 JPG/PNG 等格式</div>
          </div>
        </el-upload>
      </div>

      <!-- 编辑区/预览区 -->
      <div v-else class="preview-area">
        <div class="image-wrapper">
            <img ref="previewImg" :src="currentDisplayUrl" :style="filterStyle" alt="Preview" />
        </div>

        <!-- 编辑模式控制器 -->
        <div v-if="isEditing" class="edit-controls">
           <div class="control-group">
              <span>亮度 ({{ editState.brightness }}%)</span>
              <el-slider v-model="editState.brightness" :min="50" :max="150" :format-tooltip="val => val + '%'"></el-slider>
           </div>
           <div class="control-group">
              <span>对比度 ({{ editState.contrast }}%)</span>
              <el-slider v-model="editState.contrast" :min="50" :max="150" :format-tooltip="val => val + '%'"></el-slider>
           </div>
           <div class="control-buttons">
              <el-button round size="small" @click="rotateLeft">↺ 向左旋转</el-button>
              <el-button round size="small" @click="rotateRight">↻ 向右旋转</el-button>
           </div>
           
           <div class="action-buttons">
               <el-button type="info" plain @click="cancelEdit">取消</el-button>
               <el-button type="primary" @click="confirmEdit">确认应用</el-button>
           </div>
        </div>

        <!-- 正常模式按钮 -->
        <div v-else class="action-buttons">
            <el-button v-if="!restoredUrl && !loading" type="info" plain icon="el-icon-edit" @click="startEdit">编辑图片</el-button>
            <el-button v-if="!restoredUrl" type="primary" :loading="loading" @click="restoreImage">
              {{ loading ? '正在去雨...' : '开始恢复' }}
            </el-button>
            <el-button v-if="!loading && !restoredUrl" type="danger" plain size="small" @click="resetAll" style="margin-left:auto">重置</el-button>
        </div>
      </div>

      <!-- 进度条 -->
      <div v-if="loading" class="progress-area">
        <el-progress :percentage="progress" :status="progress === 100 ? 'success' : ''" :stroke-width="12" striped striped-flow></el-progress>
        <p class="loading-text">正在运用 AI 模型进行图像降噪...</p>
      </div>

      <!-- 结果展示 -->
      <div v-if="restoredUrl" class="result-area">
        <div class="divider">
            <span>恢复结果</span>
        </div>
        <div class="image-wrapper result-wrapper">
          <img :src="restoredUrl" alt="Restored" />
        </div>
        <div class="action-buttons">
             <el-button type="primary" class="download-btn" @click="downloadRestored">下载高清原图</el-button>
             <el-button type="text" @click="resetAll">处理下一张</el-button>
        </div>
      </div>
      
    </div>
    
    <div class="footer">
        Powered by DerainSys &copy; 2025 | 合肥工业大学梁禹设计
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

// --- State ---
const file = ref(null)            // 原始文件对象
const originalUrl = ref('')       // 原始图片URL (用于重置)
const currentDisplayUrl = ref('') // 当前展示的图片URL (可能是编辑过的)
const restoredUrl = ref('')       // 恢复后的图片URL

const loading = ref(false)
const progress = ref(0)
const isEditing = ref(false)

// 编辑状态
const editState = ref({
  brightness: 100,
  contrast: 100,
  rotation: 0 // 0, 90, 180, 270
})

// CSS 滤镜预览 (仅用于预览，不改变实际文件)
const filterStyle = computed(() => {
    if (!isEditing.value) return {}
    return {
        filter: `brightness(${editState.value.brightness}%) contrast(${editState.value.contrast}%)`,
        transform: `rotate(${editState.value.rotation}deg)`,
        transition: 'all 0.3s ease'
    }
})

// --- Methods ---

function handleFileChange(uploadFile) {
  if (!uploadFile || !uploadFile.raw) return
  // 简单的格式校验
  const isImg = uploadFile.raw.type.startsWith('image/')
  if (!isImg) {
      ElMessage.error('请上传图片文件')
      return
  }
  
  file.value = uploadFile.raw
  const url = URL.createObjectURL(uploadFile.raw)
  originalUrl.value = url
  currentDisplayUrl.value = url
  editState.value = { brightness: 100, contrast: 100, rotation: 0 }
  restoredUrl.value = ''
}

function startEdit() {
    isEditing.value = true
}

function cancelEdit() {
    isEditing.value = false
    // 重置滑块视图，但不重置文件，如果之前没确认过，那就回到原图
    editState.value = { brightness: 100, contrast: 100, rotation: 0 }
}

function rotateLeft() {
    editState.value.rotation = (editState.value.rotation - 90) % 360
}

function rotateRight() {
    editState.value.rotation = (editState.value.rotation + 90) % 360
}

// 核心：将 CSS 编辑效果“烘焙”进图片文件
async function confirmEdit() {
    try {
        const processedBlob = await processImageWithCanvas(
            currentDisplayUrl.value, 
            editState.value
        )
        // 更新当前显示
        const newUrl = URL.createObjectURL(processedBlob)
        currentDisplayUrl.value = newUrl
        // 更新待上传文件 (将 Blob 转为 File)
        file.value = new File([processedBlob], "edited_image.png", { type: "image/png" })
        
        // 重置编辑状态 (因为新图片已经是旋转/调色过的了)
        editState.value = { brightness: 100, contrast: 100, rotation: 0 }
        isEditing.value = false
        ElMessage.success('编辑已应用')
    } catch (e) {
        console.error(e)
        ElMessage.error('图片处理失败')
    }
}

// 利用 Canvas 处理图片
function processImageWithCanvas(imgSrc, settings) {
    return new Promise((resolve, reject) => {
        const img = new Image()
        img.onload = () => {
            const canvas = document.createElement('canvas')
            const ctx = canvas.getContext('2d')
            
            // 计算旋转后的尺寸
            const rad = (settings.rotation * Math.PI) / 180
            const sin = Math.abs(Math.sin(rad))
            const cos = Math.abs(Math.cos(rad))
            const width = img.width
            const height = img.height
            const newWidth = width * cos + height * sin
            const newHeight = width * sin + height * cos
            
            canvas.width = newWidth
            canvas.height = newHeight
            
            // 填充背景防止透明
            ctx.fillStyle = "#000000" 
            // 如果是PNG想要透明背景可以注释掉上面

            // 移动原点到中心以便旋转
            ctx.translate(canvas.width / 2, canvas.height / 2)
            ctx.rotate(rad)
            
            // 应用滤镜
            ctx.filter = `brightness(${settings.brightness}%) contrast(${settings.contrast}%)`
            
            // 绘制图片 (坐标偏移回中心)
            ctx.drawImage(img, -width / 2, -height / 2)
            
            canvas.toBlob((blob) => {
                if(blob) resolve(blob)
                else reject(new Error('Canvas to Blob failed'))
            }, 'image/png')
        }
        img.onerror = reject
        img.src = imgSrc
    })
}

async function restoreImage() {
  if (!file.value) return
  loading.value = true
  progress.value = 0
  restoredUrl.value = ''
  
  const formData = new FormData()
  formData.append('file', file.value)

  // 模拟进度条
  const timer = setInterval(() => {
    if (progress.value < 90) {
      progress.value += Math.floor(Math.random() * 5) + 1
    }
  }, 200)

  try {
    const response = await axios.post('/api/restore-image', formData, {
      responseType: 'blob',
      // 上传进度
      // onUploadProgress: () => {}
    })
    
    // 成功
    progress.value = 100
    restoredUrl.value = URL.createObjectURL(response.data)
  } catch (error) {
    ElMessage.error('去雨失败，请检查后端服务')
    console.error(error)
  } finally {
    clearInterval(timer)
    setTimeout(() => {
        loading.value = false
    }, 500)
  }
}

function downloadRestored() {
  if (!restoredUrl.value) return
  const a = document.createElement('a')
  a.href = restoredUrl.value
  a.download = 'restored_result.png'
  a.click()
}

function resetAll() {
    file.value = null
    originalUrl.value = ''
    currentDisplayUrl.value = ''
    restoredUrl.value = ''
    isEditing.value = false
    loading.value = false
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

.app-container {
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #fff;
}

/* 磨砂玻璃卡片 */
.glass-card {
  width: 100%;
  max-width: 500px;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  padding: 40px;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  display: flex;
  flex-direction: column;
  align-items: center;
  transition: transform 0.3s ease;
  animation: cardEntry 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
  opacity: 0;
  transform: translateY(40px);
}

@keyframes cardEntry {
  to {
      opacity: 1;
      transform: translateY(0);
  }
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0,0,0,0.5);
}

.title {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(to right, #6EE7B7, #3B82F6, #9333EA);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
}

.subtitle {
    margin: 8px 0 32px 0;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    letter-spacing: 1px;
}

/* 上传区样式 */
.upload-area {
    width: 100%;
}
.upload-demo :deep(.el-upload-dragger) {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 16px;
    width: 100%;
    height: 200px;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.3s;
}
.upload-demo :deep(.el-upload-dragger):hover {
    border-color: #3B82F6;
    background: rgba(59, 130, 246, 0.1);
}
.upload-content {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.upload-icon {
    font-size: 48px;
    color: rgba(255,255,255,0.5);
    margin-bottom: 12px;
}
.upload-text {
    font-size: 1rem;
    color: #eee;
}
.upload-hint {
    font-size: 0.8rem;
    color: #888;
    margin-top: 6px;
}

/* 预览区 */
.preview-area {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.image-wrapper {
    width: 100%;
    border-radius: 16px;
    overflow: hidden;
    background: rgba(0,0,0,0.2);
    margin-bottom: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.image-wrapper img {
    max-width: 100%;
    max-height: 400px;
    display: block;
}

/* 编辑控件 */
.edit-controls {
    width: 100%;
    background: rgba(0,0,0,0.2);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.control-group {
    margin-bottom: 12px;
}
.control-group span {
    font-size: 0.85rem;
    color: #ccc;
    display: block;
    margin-bottom: 4px;
}
.control-buttons {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 16px 0;
}
.action-buttons {
    width: 100%;
    display: flex;
    justify-content: center;
    gap: 16px;
}

/* 进度条 */
.progress-area {
    width: 100%;
    margin: 20px 0;
    text-align: center;
}
.loading-text {
    margin-top: 10px;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.7);
    animation: pulse 1.5s infinite;
}

/* 结果区 */
.result-area {
    width: 100%;
    margin-top: 20px;
    animation: fadeIn 0.5s ease-out;
}
.result-wrapper {
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    border-color: rgba(59, 130, 246, 0.4);
}
.divider {
    display: flex;
    align-items: center;
    margin-bottom: 16px;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.1);
}
.divider span {
    padding: 0 12px;
    color: #3B82F6;
    font-weight: 600;
    font-size: 0.9rem;
}

.download-btn {
    width: 100%;
    font-weight: 600;
    letter-spacing: 1px;
}

.footer {
    margin-top: 40px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.3);
}

/* 动画定义 */
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>