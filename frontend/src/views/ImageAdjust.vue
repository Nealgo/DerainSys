<template>
  <div class="flex justify-center p-8 w-full min-h-[85vh] box-border">
    <div class="w-full max-w-6xl p-8 bg-white/70 backdrop-blur-2xl border border-white/60 rounded-[2rem] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] flex flex-col items-center animate-[fadeIn_0.6s_ease-out] transition-all duration-300 hover:shadow-[0_25px_70px_-15px_rgba(0,0,0,0.15)]">
      
      <!-- Header -->
      <h1 class="text-center mb-6 text-3xl font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent tracking-tight flex items-center justify-center">
         <el-icon class="mr-3 text-indigo-600"><Setting /></el-icon> 色彩调整 (Color Adjustment)
      </h1>
      <p class="mb-10 text-slate-500 tracking-[0.2em] text-sm font-semibold uppercase opacity-80">亮度对比 · 饱和度 · 黑白效果</p>

      <!-- Upload Area -->
      <div v-if="!currentUrl" class="w-full flex justify-center py-12">
        <el-upload
          class="upload-demo group"
          drag
          action=""
          :auto-upload="false"
          :show-file-list="false"
          :on-change="handleFileChange"
        >
          <div class="relative flex flex-col items-center justify-center h-72 w-[32rem] border-2 border-dashed border-slate-300 rounded-[2rem] bg-slate-50/50 transition-all duration-500 group-hover:border-blue-500 group-hover:bg-blue-50/30 group-hover:scale-[1.02] group-hover:shadow-xl overflow-hidden">
            <div class="absolute inset-0 bg-gradient-to-tr from-blue-100/0 via-blue-100/30 to-purple-100/0 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
            <el-icon class="text-7xl text-slate-300 mb-6 transition-all duration-300 group-hover:text-blue-500 group-hover:scale-110 group-hover:-rotate-12"><UploadFilled /></el-icon>
            <div class="text-xl font-bold text-slate-600 relative z-10 transition-colors group-hover:text-blue-600">点击或拖拽上传图片</div>
            <div class="text-sm text-slate-400 mt-2 relative z-10">支持 JPG/PNG 高清原图</div>
          </div>
        </el-upload>
      </div>

      <!-- Workspace -->
      <div v-else class="w-full grid grid-cols-[2fr_1fr] gap-8 animate-[slideUp_0.5s_ease-out]">
          <!-- Image Preview -->
          <div class="bg-slate-100 rounded-2xl overflow-hidden shadow-inner border border-slate-200/60 flex items-center justify-center min-h-[400px] p-2">
              <img ref="previewImg" :src="currentUrl" :style="filterStyle" class="max-w-full max-h-[500px] object-contain rounded-xl" />
          </div>

          <!-- Controls -->
          <div class="bg-white/50 backdrop-blur-sm rounded-2xl p-6 border border-white/60 shadow-sm flex flex-col gap-6">
               <div class="flex items-center justify-between">
                   <span class="text-sm font-bold text-slate-500 uppercase tracking-wider">黑白模式 (Grayscale)</span>
                   <el-switch v-model="settings.grayscale" active-color="#4f46e5" />
               </div>
               
               <div>
                   <div class="flex justify-between mb-2">
                       <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Brightness</span>
                       <span class="text-xs font-mono text-blue-600">{{ settings.brightness }}%</span>
                   </div>
                   <el-slider v-model="settings.brightness" :min="0" :max="200" />
               </div>

               <div>
                   <div class="flex justify-between mb-2">
                       <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Contrast</span>
                       <span class="text-xs font-mono text-blue-600">{{ settings.contrast }}%</span>
                   </div>
                   <el-slider v-model="settings.contrast" :min="0" :max="200" />
               </div>
               
               <div>
                   <div class="flex justify-between mb-2">
                       <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Saturation</span>
                       <span class="text-xs font-mono text-blue-600">{{ settings.saturate }}%</span>
                   </div>
                   <el-slider v-model="settings.saturate" :min="0" :max="200" />
               </div>

               <div class="mt-auto flex flex-col gap-3">
                   <button class="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-bold shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-[1.02] active:scale-95 transition-all flex items-center justify-center" @click="downloadResult">
                       <el-icon class="mr-2"><Download /></el-icon> 下载结果
                   </button>
                   <button class="w-full py-2.5 rounded-xl text-slate-500 hover:bg-slate-100 hover:text-red-500 transition-colors flex items-center justify-center" @click="reset">
                       <el-icon class="mr-2"><Delete /></el-icon> 重置
                   </button>
               </div>
          </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { UploadFilled, Download, Delete, Setting } from '@element-plus/icons-vue'

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
