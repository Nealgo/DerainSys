<template>
  <div class="flex justify-center p-4 w-full h-[calc(100vh-80px)] box-border">
    <div class="w-full max-w-6xl p-6 bg-white/70 backdrop-blur-2xl border border-white/60 rounded-[2rem] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] flex flex-col animate-[fadeIn_0.6s_ease-out] transition-all duration-300 hover:shadow-[0_25px_70px_-15px_rgba(0,0,0,0.15)] h-full">
      
      <!-- Header -->
      <h1 class="text-center mb-6 text-3xl font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent tracking-tight flex items-center justify-center">
         <el-icon class="mr-3 text-indigo-600"><Picture /></el-icon> 图片拼接 (Image Stitching)
      </h1>
      <p class="text-center mb-4 text-slate-500 tracking-[0.2em] text-sm font-semibold uppercase opacity-80">多图合一 · 竖向横向 · 批量处理</p>

      <div class="grid grid-cols-[280px_1fr] gap-4 w-full flex-1 min-h-0">
        <!-- Left Panel: Upload & List -->
        <div class="flex flex-col gap-3 p-4 bg-white/50 rounded-2xl border border-white/60 shadow-sm overflow-hidden">
            <el-upload
                class="stitch-upload"
                action=""
                multiple
                :auto-upload="false"
                :show-file-list="false"
                :on-change="handleFileChange"
            >
               <button class="w-full py-2.5 text-sm rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-md shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-[1.02] active:scale-95 transition-all flex items-center justify-center">
                   <el-icon class="mr-1.5 text-base"><Plus /></el-icon> 添加图片
               </button>
            </el-upload>
            
            <div class="flex-1 overflow-y-auto space-y-2 min-h-0" v-if="imageList.length > 0">
                <div v-for="(img, index) in imageList" :key="img.id" class="flex items-center gap-2 p-2 bg-slate-50 rounded-lg border border-slate-100 group hover:border-blue-200 hover:bg-blue-50/50 transition-all">
                    <div class="w-10 h-10 rounded-md overflow-hidden flex-shrink-0 border border-slate-200">
                        <img :src="img.url" class="w-full h-full object-cover" />
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs font-medium text-slate-600 truncate">{{ img.file.name }}</p>
                        <p class="text-[10px] text-slate-400">{{ (img.file.size / 1024).toFixed(1) }} KB</p>
                    </div>
                    <div class="flex gap-1 opacity-50 group-hover:opacity-100 transition-opacity">
                        <button class="w-6 h-6 rounded-full bg-slate-100 text-slate-500 hover:bg-blue-100 hover:text-blue-600 transition-colors flex items-center justify-center disabled:opacity-30 text-xs" @click="moveUp(index)" :disabled="index === 0"><el-icon><Top /></el-icon></button>
                        <button class="w-6 h-6 rounded-full bg-slate-100 text-slate-500 hover:bg-blue-100 hover:text-blue-600 transition-colors flex items-center justify-center disabled:opacity-30 text-xs" @click="moveDown(index)" :disabled="index === imageList.length - 1"><el-icon><Bottom /></el-icon></button>
                        <button class="w-6 h-6 rounded-full bg-red-50 text-red-400 hover:bg-red-500 hover:text-white transition-colors flex items-center justify-center text-xs" @click="removeImage(index)"><el-icon><Delete /></el-icon></button>
                    </div>
                </div>
            </div>
            <div v-else class="flex-1 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-200 rounded-xl min-h-[280px]">
                <el-icon class="text-4xl mb-2"><Picture /></el-icon>
                <p class="text-xs">暂无图片，请添加</p>
            </div>
        </div>

        <!-- Right Panel: Settings & Preview -->
        <div class="flex flex-col gap-4 p-4 bg-white/50 rounded-2xl border border-white/60 shadow-sm overflow-hidden">
            <div class="flex items-center justify-center gap-4 flex-shrink-0 py-2 border-b border-slate-100">
                <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">拼接方向</span>
                <el-radio-group v-model="stitchMode" size="default" class="bg-gradient-to-r from-slate-50 to-slate-100 p-1 rounded-full shadow-inner border border-slate-200/50">
                    <el-radio-button label="vertical" class="!rounded-full !border-none !shadow-none !text-xs !px-4">
                         <el-icon class="mr-1"><Bottom /></el-icon> 竖向
                    </el-radio-button>
                    <el-radio-button label="horizontal" class="!rounded-full !border-none !shadow-none !text-xs !px-4">
                         <el-icon class="mr-1"><Right /></el-icon> 横向
                    </el-radio-button>
                </el-radio-group>
            </div>

            <div class="flex-1 bg-slate-100 rounded-xl overflow-auto shadow-inner border border-slate-200/60 flex items-center justify-center min-h-0">
                 <div class="p-4" v-if="previewUrl">
                     <img :src="previewUrl" class="max-w-full max-h-[400px] block shadow-lg rounded-lg" />
                 </div>
                 <div v-else class="text-slate-400 text-xs">
                     <span>点击"开始拼接"生成预览</span>
                 </div>
            </div>

            <div class="flex gap-3 flex-shrink-0">
                <button class="group relative flex-1 py-3.5 text-sm rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:shadow-xl hover:scale-[1.03] hover:-translate-y-0.5 active:scale-95 transition-all duration-300 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden" :disabled="imageList.length < 2" @click="processStitch">
                    <span class="absolute inset-0 bg-gradient-to-r from-white/0 via-white/30 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out"></span>
                    <el-icon class="mr-2 group-hover:animate-pulse"><MagicStick /></el-icon> 开始拼接
                </button>
                <button class="group relative flex-1 py-3.5 text-sm rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 hover:shadow-xl hover:scale-[1.03] hover:-translate-y-0.5 active:scale-95 transition-all duration-300 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed overflow-hidden" :disabled="!previewUrl" @click="downloadResult">
                    <span class="absolute inset-0 bg-gradient-to-r from-white/0 via-white/30 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700 ease-in-out"></span>
                    <el-icon class="mr-2 group-hover:animate-bounce"><Download /></el-icon> 下载结果
                </button>
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
    
    try {
        const loadedImgs = await Promise.all(imageList.value.map(item => loadImage(item.url)))
        
        let totalWidth = 0
        let totalHeight = 0
        
        if(stitchMode.value === 'vertical') {
            totalWidth = Math.max(...loadedImgs.map(img => img.width))
            totalHeight = loadedImgs.reduce((sum, img) => sum + img.height, 0)
        } else {
            totalWidth = loadedImgs.reduce((sum, img) => sum + img.width, 0)
            totalHeight = Math.max(...loadedImgs.map(img => img.height))
        }
        
        const canvas = document.createElement('canvas')
        canvas.width = totalWidth
        canvas.height = totalHeight
        const ctx = canvas.getContext('2d')

        let currentX = 0
        let currentY = 0
        
        loadedImgs.forEach(img => {
            if(stitchMode.value === 'vertical') {
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
            ElMessage.success('拼接完成')
        }, 'image/png')
        
    } catch (e) {
        console.error(e)
        ElMessage.error('处理出错')
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
.stitch-upload {
  width: 100%;
}
.stitch-upload :deep(.el-upload) {
  width: 100%;
}
</style>
