<template>
  <div class="flex justify-center p-4 w-full h-[calc(100vh-80px)] box-border">
    <div class="w-full max-w-6xl p-6 bg-white/70 backdrop-blur-2xl border border-white/60 rounded-[2rem] shadow-[0_20px_60px_-15px_rgba(0,0,0,0.1)] flex flex-col animate-[fadeIn_0.6s_ease-out] transition-all duration-300 hover:shadow-[0_25px_70px_-15px_rgba(0,0,0,0.15)] h-full overflow-hidden">
      
      <!-- Header -->
      <h1 class="text-center mb-6 text-3xl font-black bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600 bg-clip-text text-transparent tracking-tight flex items-center justify-center">
         <el-icon class="mr-3 text-indigo-600"><Crop /></el-icon> 智能裁剪 (Smart Cropping)
      </h1>
      <p class="text-center mb-4 text-slate-500 tracking-[0.2em] text-sm font-semibold uppercase opacity-80">多模式裁剪 · 自由形状 · 实时预览</p>

      <!-- Step 1: Upload -->
      <div v-if="!imgUrl" class="w-full flex justify-center py-12">
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

      <!-- Step 2: Crop Workspace -->
      <div v-else class="w-full flex flex-col flex-1 min-h-0 overflow-hidden">
        <!-- Mode Switcher -->
        <div class="flex justify-center mb-4 flex-shrink-0">
           <el-radio-group v-model="cropMode" size="large" class="bg-slate-100 p-1 rounded-full shadow-inner">
              <el-radio-button label="rect" class="!rounded-full !border-none !shadow-none"><el-icon class="mr-1"><Crop /></el-icon> 矩形裁剪 (Rect)</el-radio-button>
              <el-radio-button label="poly" class="!rounded-full !border-none !shadow-none"><el-icon class="mr-1"><EditPen /></el-icon> 自由形状 (Polygon)</el-radio-button>
           </el-radio-group>
        </div>

        <div class="grid grid-cols-[2fr_1fr] gap-4 flex-1 min-h-0 overflow-hidden">
            <!-- Left: Workspace -->
            <div class="flex flex-col gap-4">
              <div class="flex-1 bg-slate-900 rounded-2xl overflow-hidden shadow-inner border border-slate-700/50 relative">
                <!-- Rectangular Mode -->
                <VueCropper
                  v-show="cropMode === 'rect'"
                  ref="cropperRef"
                  :img="imgUrl"
                  :outputSize="option.outputSize"
                  :outputType="option.outputType"
                  :info="true"
                  :full="option.full"
                  :canMove="option.canMove"
                  :canMoveBox="option.canMoveBox"
                  :original="option.original"
                  :autoCrop="option.autoCrop"
                  :autoCropWidth="option.autoCropWidth"
                  :autoCropHeight="option.autoCropHeight"
                  :fixedBox="option.fixedBox"
                  @realTime="realTime"
                />
                
                <!-- Polygon Mode -->
                <div v-show="cropMode === 'poly'" class="w-full h-full flex items-center justify-center cursor-crosshair" ref="polyWrapper">
                    <canvas ref="polyCanvas" @mousedown="handleCanvasClick" @mousemove="handleCanvasMove"></canvas>
                </div>
              </div>

              <div class="text-center text-sm text-slate-500 font-medium py-2 bg-slate-50/50 rounded-lg">
                <template v-if="cropMode === 'rect'">
                    <el-icon class="align-middle mr-1"><InfoFilled /></el-icon> 滚动鼠标缩放，拖动选框裁剪
                </template>
                <template v-else>
                    <el-icon class="align-middle mr-1"><InfoFilled /></el-icon> 点击画布添加锚点，连接成多边形。双击或闭合路径完成。
                </template>
              </div>
            </div>

            <!-- Right: Controls & Preview -->
            <div class="flex flex-col gap-4 overflow-y-auto">
              <!-- Preview Section -->
              <div class="bg-white/50 backdrop-blur-sm rounded-2xl p-4 border border-white/60 shadow-sm">
                 <h3 class="w-full text-left text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 border-b border-slate-200 pb-2 flex items-center"><el-icon class="mr-1"><View /></el-icon> 实时预览</h3>
                 <div class="w-full h-40 bg-[url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHUlEQVQ4jWNgYGAQIYJjY8D8Q0NnNKGj0YSOxhMAi8EBCeD3f6gAAAAASUVORK5CYII=')] rounded-lg overflow-hidden shadow-inner flex items-center justify-center border border-slate-200">
                    <!-- Rect Preview - using actual cropped data -->
                    <img v-if="cropMode === 'rect' && rectPreviewUrl" :src="rectPreviewUrl" class="max-w-full max-h-full object-contain" />
                    <span v-else-if="cropMode === 'rect'" class="text-xs text-slate-400">调整选框查看预览</span>
                    <!-- Poly Preview -->
                    <div v-else class="w-full h-full flex items-center justify-center p-2">
                        <img v-if="polyPreviewUrl" :src="polyPreviewUrl" class="max-w-full max-h-full object-contain" />
                        <span v-else class="text-xs text-slate-400">绘制以预览</span>
                    </div>
                 </div>
              </div>

              <!-- Controls Section -->
              <div class="bg-white/50 backdrop-blur-sm rounded-2xl p-4 border border-white/60 shadow-sm flex flex-col gap-3">
                <div v-if="cropMode === 'rect'" class="flex justify-center gap-3">
                    <button class="w-9 h-9 rounded-full border border-slate-300 text-slate-500 hover:bg-blue-50 hover:text-blue-600 hover:border-blue-200 transition-all flex items-center justify-center" @click="rotateLeft" title="向左旋转"><el-icon><RefreshLeft /></el-icon></button>
                    <button class="w-9 h-9 rounded-full border border-slate-300 text-slate-500 hover:bg-blue-50 hover:text-blue-600 hover:border-blue-200 transition-all flex items-center justify-center" @click="rotateRight" title="向右旋转"><el-icon><RefreshRight /></el-icon></button>
                </div>
                <div v-else class="flex justify-center gap-2">
                    <button class="px-3 py-1.5 rounded-lg bg-red-50 text-red-500 text-xs font-bold hover:bg-red-100 transition-colors" @click="clearPolyPoints">清空锚点</button>
                    <button class="px-3 py-1.5 rounded-lg bg-green-50 text-green-600 text-xs font-bold hover:bg-green-100 transition-colors" @click="closePolyPath">闭合路径</button>
                </div>
                
                <button class="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white text-sm font-bold shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-[1.02] active:scale-95 transition-all flex items-center justify-center" @click="confirmCrop">
                    <el-icon class="mr-1.5"><Download /></el-icon> 确认裁剪下载
                </button>
                <button class="w-full py-2 rounded-xl text-slate-500 text-xs hover:bg-slate-100 hover:text-red-500 transition-colors flex items-center justify-center" @click="reset">
                    <el-icon class="mr-1"><Delete /></el-icon> 重新上传
                </button>
              </div>
            </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch, nextTick } from 'vue'
import { VueCropper } from 'vue-cropper'
import 'vue-cropper/dist/index.css'
import { UploadFilled, InfoFilled, View, RefreshLeft, RefreshRight, Download, Delete, Crop, EditPen } from '@element-plus/icons-vue'

const imgUrl = ref('')
const cropMode = ref('rect') // 'rect' | 'poly'
const cropperRef = ref(null)

// Rect Preview
const preView = ref({})
const rectPreviewUrl = ref('')

// Poly Logic
const polyCanvas = ref(null)
const polyWrapper = ref(null)
const polyPoints = ref([])
const polyPreviewUrl = ref('')
let polyCtx = null
let imageObj = null
let canvasScale = 1

const option = reactive({
  img: '', 
  outputSize: 1, 
  outputType: 'png',
  info: true, 
  full: false, 
  canMove: true, 
  canMoveBox: true, 
  original: false, 
  autoCrop: true,
  autoCropWidth: 400,
  autoCropHeight: 400,
  fixedBox: false
})

function handleFileChange(uploadFile) {
  if(!uploadFile.raw.type.startsWith('image/')) return
  const reader = new FileReader()
  reader.onload = (e) => {
    imgUrl.value = e.target.result
    initPolyImage(e.target.result)
  }
  reader.readAsDataURL(uploadFile.raw)
}

function initPolyImage(src) {
    imageObj = new Image()
    imageObj.src = src
    imageObj.onload = () => {
        // Init canvas dimension
        if(cropMode.value === 'poly') {
            nextTick(() => drawPolyCanvas())
        }
    }
}

watch(cropMode, (val) => {
    if(val === 'poly' && imgUrl.value) {
        nextTick(() => drawPolyCanvas())
    }
})

// === Rect Logic ===
let previewTimeout = null
function realTime(data) {
  preView.value = data
  // Debounce getting the cropped preview to avoid too many calls
  if (previewTimeout) clearTimeout(previewTimeout)
  previewTimeout = setTimeout(() => {
    if (cropperRef.value) {
      cropperRef.value.getCropData((dataUrl) => {
        rectPreviewUrl.value = dataUrl
      })
    }
  }, 100)
}
function rotateLeft() { cropperRef.value.rotateLeft() }
function rotateRight() { cropperRef.value.rotateRight() }

// === Poly Logic ===
function drawPolyCanvas() {
    if(!polyCanvas.value || !imageObj) return
    const canvas = polyCanvas.value
    const wrapper = polyWrapper.value
    
    // Resize canvas to fit wrapper but maintain aspect ratio
    const maxWidth = wrapper.clientWidth
    const maxHeight = wrapper.clientHeight
    
    const scaleW = maxWidth / imageObj.width
    const scaleH = maxHeight / imageObj.height
    canvasScale = Math.min(scaleW, scaleH, 1) // Don't upscale
    
    canvas.width = imageObj.width * canvasScale
    canvas.height = imageObj.height * canvasScale
    
    polyCtx = canvas.getContext('2d')
    renderPolyState()
}

function renderPolyState() {
    if(!polyCtx || !imageObj) return
    // Clear
    polyCtx.clearRect(0, 0, polyCanvas.value.width, polyCanvas.value.height)
    
    // Draw Image
    polyCtx.fillStyle = 'rgba(0,0,0,0.5)'
    polyCtx.fillRect(0, 0, polyCanvas.value.width, polyCanvas.value.height)
    
    // Reset to see image clearly
    polyCtx.clearRect(0, 0, polyCanvas.value.width, polyCanvas.value.height)
    polyCtx.drawImage(imageObj, 0, 0, polyCanvas.value.width, polyCanvas.value.height)

    // Draw lines
    if(polyPoints.value.length > 0) {
        polyCtx.beginPath()
        polyCtx.lineWidth = 2
        polyCtx.strokeStyle = '#00ff00'
        polyCtx.fillStyle = 'rgba(0, 255, 0, 0.3)'
        
        const start = polyPoints.value[0]
        polyCtx.moveTo(start.x, start.y)
        
        for(let i=1; i<polyPoints.value.length; i++) {
            polyCtx.lineTo(polyPoints.value[i].x, polyPoints.value[i].y)
        }
        
        polyCtx.stroke()
        
        // Draw points
        polyCtx.fillStyle = 'white'
        for(let p of polyPoints.value) {
              polyCtx.beginPath()
              polyCtx.arc(p.x, p.y, 4, 0, Math.PI*2)
              polyCtx.fill()
        }
    }
}

function handleCanvasClick(e) {
    if(cropMode.value !== 'poly') return
    const rect = polyCanvas.value.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    polyPoints.value.push({x, y})
    renderPolyState()
}

function handleCanvasMove() {
    // Implement hover effects if needed
}

function clearPolyPoints() {
    polyPoints.value = []
    renderPolyState()
    polyPreviewUrl.value = ''
}

function closePolyPath() {
    if(polyPoints.value.length < 3) return
    
    const start = polyPoints.value[0]
    polyPoints.value.push({...start}) // Close it
    renderPolyState()
    
    generatePolyCropBlob((blob) => {
        polyPreviewUrl.value = URL.createObjectURL(blob)
    })
}

// === Common ===
function confirmCrop() {
    if(cropMode.value === 'rect') {
        cropperRef.value.getCropBlob((data) => {
            downloadBlob(data)
        })
    } else {
        generatePolyCropBlob((blob) => {
             downloadBlob(blob)
        })
    }
}

function generatePolyCropBlob(callback) {
    if(!imageObj || polyPoints.value.length < 3) return
    
    const offCanvas = document.createElement('canvas')
    offCanvas.width = imageObj.width
    offCanvas.height = imageObj.height
    const ctx = offCanvas.getContext('2d')
    
    // 1. Draw Path
    ctx.beginPath()
    const p0 = polyPoints.value[0]
    ctx.moveTo(p0.x / canvasScale, p0.y / canvasScale)
    
    for(let i=1; i<polyPoints.value.length; i++) {
         const p = polyPoints.value[i]
         ctx.lineTo(p.x / canvasScale, p.y / canvasScale)
    }
    ctx.closePath()
    
    // 2. Clip
    ctx.clip()
    
    // 3. Draw Image
    ctx.drawImage(imageObj, 0, 0)
    
    // 4. Export
    offCanvas.toBlob(callback)
}

function downloadBlob(blob) {
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cropped_${Date.now()}.png`
    a.click()
}

function reset() {
  imgUrl.value = ''
  polyPoints.value = []
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
/* Preview wrapper styling */
.preview-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
