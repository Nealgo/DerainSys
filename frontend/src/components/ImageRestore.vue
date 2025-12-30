<template>
  <div class="flex justify-center p-8 w-full min-h-[85vh] box-border">
    <div class="w-full max-w-6xl p-10 bg-white/70 backdrop-blur-2xl border border-white/60 rounded-[2rem] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] flex flex-col items-center animate-[fadeIn_0.6s_ease-out] transition-all duration-300 hover:shadow-[0_25px_70px_-15px_rgba(0,0,0,0.15)]">
      
      <!-- Header -->
      <h1 class="text-center mb-6 text-4xl font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent tracking-tight leading-tight drop-shadow-sm">
        <el-icon class="align-middle mr-3 text-indigo-600 animate-pulse"><MagicStick /></el-icon> AI Image Deraining
      </h1>
      <p class="mb-12 text-slate-500 tracking-[0.2em] text-sm font-semibold uppercase opacity-80">智能去雨 · 细节还原 · 极致体验</p>

      <!-- 上传区域 -->
      <div v-if="!originalUrl" class="w-full flex justify-center py-10">
        <el-upload
          class="upload-demo group"
          drag
          action=""
          :auto-upload="false"
          :show-file-list="false"
          :on-change="handleFileChange"
        >
          <div class="relative flex flex-col items-center justify-center h-72 w-[32rem] border-2 border-dashed border-slate-300 rounded-[2rem] bg-slate-50/50 transition-all duration-500 group-hover:border-blue-500 group-hover:bg-blue-50/30 group-hover:scale-[1.02] group-hover:shadow-2xl overflow-hidden">
             <!-- Animated background effect on hover -->
             <div class="absolute inset-0 bg-gradient-to-tr from-blue-100/0 via-blue-100/30 to-purple-100/0 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
            
            <el-icon class="text-7xl text-slate-300 mb-6 transition-all duration-300 group-hover:text-blue-500 group-hover:scale-110 group-hover:-rotate-12"><UploadFilled /></el-icon>
            <div class="text-xl font-bold text-slate-600 relative z-10 transition-colors group-hover:text-blue-600">点击或拖拽上传图片</div>
            <div class="text-sm text-slate-400 mt-2 relative z-10">支持 JPG/PNG 高清原图</div>
          </div>
        </el-upload>
      </div>

      <!-- 结果对比展示 (Side-by-Side) -->
      <div v-else-if="restoredUrl" class="w-full flex flex-col items-center animate-[scaleIn_0.5s_ease-out]">
         <div class="flex flex-wrap gap-10 w-full justify-center">
            <!-- Original -->
            <div class="flex-1 min-w-[320px] flex flex-col items-center group">
               <div class="relative w-full rounded-2xl overflow-hidden shadow-xl border-4 border-white bg-slate-100 transition-transform duration-500 group-hover:scale-[1.01] group-hover:shadow-2xl">
                  <div class="absolute top-4 left-4 px-4 py-1.5 rounded-full text-xs font-bold tracking-wider text-white bg-slate-900/40 backdrop-blur-md border border-white/20 z-10 shadow-sm">BEFORE</div>
                  <img :src="currentDisplayUrl" alt="Original" class="w-full h-auto block aspect-[4/3] object-contain bg-[url('https://gin-vue-admin.com/assets/images/tile.png')] bg-[length:20px_20px]" />
               </div>
            </div>
            
            <!-- Restored -->
            <div class="flex-1 min-w-[320px] flex flex-col items-center group">
               <div class="relative w-full rounded-2xl overflow-hidden shadow-[0_0_40px_rgba(79,70,229,0.15)] border-4 border-white ring-4 ring-indigo-50 bg-slate-100 transition-transform duration-500 group-hover:scale-[1.01] group-hover:shadow-[0_0_60px_rgba(79,70,229,0.3)]">
                  <div class="absolute top-4 left-4 px-4 py-1.5 rounded-full text-xs font-bold tracking-wider text-white bg-gradient-to-r from-emerald-500/80 to-green-500/80 backdrop-blur-md border border-white/20 z-10 shadow-sm">AFTER</div>
                  <img :src="restoredUrl" alt="Restored" class="w-full h-auto block aspect-[4/3] object-contain bg-[url('https://gin-vue-admin.com/assets/images/tile.png')] bg-[length:20px_20px]" />
               </div>
            </div>
         </div>

         <div class="flex justify-center gap-6 mt-12 w-full">
             <button class="px-8 py-3.5 rounded-full bg-slate-100 text-slate-600 font-bold tracking-wide border border-slate-200 shadow-sm hover:bg-white hover:text-red-500 hover:border-red-200 hover:shadow-md transition-all duration-300 flex items-center group" @click="resetAll">
                <el-icon class="mr-2 text-lg transition-transform group-hover:-rotate-180"><RefreshLeft /></el-icon> 处理下一张
             </button>
             <button class="px-10 py-3.5 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold tracking-wide shadow-lg hover:shadow-blue-500/40 hover:-translate-y-1 hover:brightness-110 active:scale-95 transition-all duration-300 flex items-center" @click="downloadRestored">
                <el-icon class="mr-2 text-xl font-bold"><Download /></el-icon> 下载高清结果
             </button>
         </div>
      </div>

      <!-- 编辑区/预览区 (Pre-Processing) -->
      <div v-else class="w-full flex flex-col items-center overflow-hidden">
        <div class="w-full rounded-2xl overflow-hidden bg-slate-100 mb-6 flex justify-center items-center border border-slate-200/60 shadow-inner p-2 max-w-3xl" style="min-height: 300px; max-height: 50vh;">
            <img ref="previewImg" :src="currentDisplayUrl" :style="filterStyle" alt="Preview" class="max-w-full max-h-[45vh] object-contain rounded-xl" style="image-rendering: auto;" />
        </div>

        <!-- 编辑模式控制器 -->
        <div v-if="isEditing" class="w-full max-w-2xl bg-white/60 backdrop-blur-xl p-8 rounded-[2rem] mb-8 shadow-lg border border-white/50 animate-[slideUp_0.4s_cubic-bezier(0.16,1,0.3,1)]">
           <div class="mb-6 space-y-6">
               <div>
                  <div class="flex justify-between mb-2">
                      <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Brightness</span>
                      <span class="text-xs font-mono text-blue-600">{{ editState.brightness }}%</span>
                  </div>
                  <el-slider v-model="editState.brightness" :min="50" :max="150" :format-tooltip="val => val + '%'"></el-slider>
               </div>
               <div>
                  <div class="flex justify-between mb-2">
                       <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Contrast</span>
                       <span class="text-xs font-mono text-blue-600">{{ editState.contrast }}%</span>
                  </div>
                  <el-slider v-model="editState.contrast" :min="50" :max="150" :format-tooltip="val => val + '%'"></el-slider>
               </div>
           </div>
           
           <div class="flex justify-center gap-6 my-6 pb-6 border-b border-slate-100">
              <button class="w-12 h-12 rounded-full bg-slate-50 border border-slate-200 text-slate-500 hover:bg-white hover:text-blue-600 hover:shadow-md hover:border-blue-100 transition-all active:scale-90 flex items-center justify-center text-xl" @click="rotateLeft" title="Rotate Left">↺</button>
              <button class="w-12 h-12 rounded-full bg-slate-50 border border-slate-200 text-slate-500 hover:bg-white hover:text-blue-600 hover:shadow-md hover:border-blue-100 transition-all active:scale-90 flex items-center justify-center text-xl" @click="rotateRight" title="Rotate Right">↻</button>
           </div>
           
           <div class="flex justify-center gap-4 w-full">
               <button class="px-8 py-2.5 rounded-full text-slate-500 font-semibold hover:bg-slate-100 transition-colors" @click="cancelEdit">Cancel</button>
               <button class="px-8 py-2.5 rounded-full bg-slate-900 text-white font-semibold hover:bg-slate-700 hover:scale-105 transition-all shadow-lg" @click="confirmEdit">Apply Changes</button>
           </div>
        </div>

        <!-- 正常模式按钮 (fixed min-height to prevent layout shift) -->
        <div v-if="!isEditing" class="min-h-[64px] flex items-center justify-center gap-6 w-full">
             <button v-if="!loading" class="px-6 py-3 rounded-full bg-white text-slate-600 border border-slate-200 font-bold shadow-sm hover:shadow-md hover:border-blue-300 hover:text-blue-600 transition-all duration-300 flex items-center" @click="startEdit">
                <el-icon class="mr-2"><Edit /></el-icon> 编辑图片
            </button>
            <button class="relative overflow-hidden px-12 py-3 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold tracking-wider shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-105 active:scale-95 transition-all duration-300 flex items-center disabled:opacity-70 disabled:cursor-not-allowed group" :disabled="loading" @click="restoreImage">
              <span v-if="!loading" class="absolute inset-0 bg-white/20 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out"></span>
              <el-icon v-if="!loading" class="mr-2 text-lg"><MagicStick /></el-icon>
              {{ loading ? 'Generating...' : '开始恢复 (Start Restore)' }}
            </button>
            <button v-if="!loading" class="w-12 h-12 rounded-full bg-red-50 text-red-400 border border-red-100 hover:bg-red-500 hover:text-white hover:shadow-lg hover:shadow-red-500/30 transition-all active:scale-90 flex items-center justify-center" @click="resetAll">
                <el-icon><Refresh /></el-icon>
            </button>
        </div>

        <!-- 进度条 -->
        <div v-if="loading" class="w-full max-w-lg mt-10 text-center">
          <div class="h-3 w-full bg-slate-100 rounded-full overflow-hidden">
             <div class="h-full bg-gradient-to-r from-blue-400 to-indigo-600 animate-[progress_2s_ease-in-out_infinite]" :style="{ width: progress + '%' }"></div>
          </div>
          <p class="mt-4 text-sm font-semibold text-slate-400 animate-pulse tracking-wide">AI Engine Processing... Please Wait</p>
        </div>
      </div>
      
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { UploadFilled, Edit, MagicStick, Refresh, Download, RefreshLeft } from '@element-plus/icons-vue'

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
/* Override el-upload default styling */
:deep(.el-upload-dragger) {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  width: auto !important;
  height: auto !important;
}
:deep(.el-upload) {
  width: 100%;
}
</style>