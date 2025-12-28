<template>
  <div class="page-container">
    <div class="glass-card main-card">
      <h1 class="page-title">✨ 智能裁剪 (Smart Cropping)</h1>

      <!-- Step 1: Upload -->
      <div v-if="!imgUrl" class="upload-area">
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

      <!-- Step 2: Crop Workspace -->
      <div v-else class="workspace">
        <!-- Mode Switcher -->
        <div class="mode-switch-container">
           <el-radio-group v-model="cropMode" size="large">
              <el-radio-button label="rect"><el-icon><Crop /></el-icon> 矩形裁剪 (Rect)</el-radio-button>
              <el-radio-button label="poly"><el-icon><EditPen /></el-icon> 自由形状 (Polygon)</el-radio-button>
           </el-radio-group>
        </div>

        <div class="main-content-row">
            <div class="cropper-container">
              <div class="cropper-wrapper">
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
                <div v-show="cropMode === 'poly'" class="poly-canvas-wrapper" ref="polyWrapper">
                    <canvas ref="polyCanvas" @mousedown="handleCanvasClick" @mousemove="handleCanvasMove"></canvas>
                </div>
              </div>

              <div class="instruction-text">
                <template v-if="cropMode === 'rect'">
                    <el-icon><InfoFilled /></el-icon> 滚动鼠标缩放，拖动选框裁剪
                </template>
                <template v-else>
                    <el-icon><InfoFilled /></el-icon> 点击画布添加锚点，连接成多边形。双击或闭合路径完成。
                </template>
              </div>
            </div>

            <div class="controls-panel">
              <div class="preview-card">
                 <h3><el-icon><View /></el-icon> 实时预览 (Preview)</h3>
                 <div class="preview-viewport">
                    <!-- Rect Preview -->
                    <div v-if="cropMode === 'rect'" :style="preViewStyle">
                      <div :style="preView.div">
                        <img :src="preView.url" :style="preView.img">
                      </div>
                    </div>
                    <!-- Poly Preview (Simplified) -->
                    <div v-else class="poly-preview">
                        <img v-if="polyPreviewUrl" :src="polyPreviewUrl" />
                        <span v-else>绘制以预览</span>
                    </div>
                 </div>
              </div>

              <div class="action-card">
                <div v-if="cropMode === 'rect'" class="rotate-group">
                    <el-button class="tool-btn" @click="rotateLeft" circle :icon="RefreshLeft" title="左旋转"></el-button>
                    <el-button class="tool-btn" @click="rotateRight" circle :icon="RefreshRight" title="右旋转"></el-button>
                </div>
                <div v-else class="rotate-group">
                    <el-button class="tool-btn" @click="clearPolyPoints" icon="el-icon-delete">清空锚点</el-button>
                    <el-button class="tool-btn" @click="closePolyPath" icon="el-icon-check">闭合路径</el-button>
                </div>
                
                <el-button class="action-btn primary-btn" type="primary" @click="confirmCrop">
                    <el-icon style="margin-right: 8px"><Download /></el-icon> 确认裁剪并下载
                </el-button>
                <el-button class="action-btn reset-btn" type="text" @click="reset">
                    <el-icon style="margin-right: 8px"><Delete /></el-icon> 重新上传
                </el-button>
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
const preViewStyle = ref({})

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
function realTime(data) {
  preView.value = data
  preViewStyle.value = {
    width: data.w + "px",
    height: data.h + "px",
    overflow: "hidden",
    margin: "0",
    zoom: 200 / data.w 
  }
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
    
    // Clip Path (Simulated by clearing the poly area or just drawing lines)
    // Actually, improved UX: Draw standard image, then draw polygon lines on top
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

// Optional: Dragging logic omitted for brevity, adding click-to-add
function handleCanvasMove() {
    // Implement hover effects if needed
}

function clearPolyPoints() {
    polyPoints.value = []
    renderPolyState()
    polyPreviewUrl.value = ''
}

function closePolyPath() {
    // Just visual feedback, currently 'confirm' does the clipping
    if(polyPoints.value.length < 3) return
    
    // Draw closed shape
    const start = polyPoints.value[0]
    polyPoints.value.push({...start}) // Close it
    renderPolyState()
    
    // Generate Preview
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
    
    // Create an offscreen canvas with ORIGINAL dimensions
    const offCanvas = document.createElement('canvas')
    offCanvas.width = imageObj.width
    offCanvas.height = imageObj.height
    const ctx = offCanvas.getContext('2d')
    
    // 1. Draw Path
    ctx.beginPath()
    const p0 = polyPoints.value[0]
    // Map scaled coordinates back to original
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
/* Reuse existing styles plus: */
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
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    animation: fadeIn 0.6s ease-out;
    display: flex;
    flex-direction: column;
}

.mode-switch-container {
    display: flex;
    justify-content: center;
    margin-bottom: 24px;
}

.main-content-row {
     display: grid;
     grid-template-columns: 2fr 1fr;
     gap: 30px;
     height: 600px;
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
    /* Main container inside card */
    display: flex;
    flex-direction: column;
}

/* Cropper/Canvas Side */
.cropper-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 100%;
}
.cropper-wrapper {
    flex: 1;
    background: #1a1a1a;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.1);
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
}
.poly-canvas-wrapper {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: crosshair;
}

.instruction-text {
    font-size: 13px;
    color: rgba(255,255,255,0.6);
    text-align: center;
}

/* Controls Side */
.controls-panel {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.preview-card {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    border: 1px solid rgba(255,255,255,0.05);
}
.preview-card h3 {
    margin: 0 0 16px 0;
    font-size: 16px;
    color: #e0e7ff;
    width: 100%;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 10px;
}
.preview-viewport {
    width: 200px;
    height: 200px;
    background-image: linear-gradient(45deg, #2a2a2a 25%, transparent 25%), 
                      linear-gradient(-45deg, #2a2a2a 25%, transparent 25%), 
                      linear-gradient(45deg, transparent 75%, #2a2a2a 75%), 
                      linear-gradient(-45deg, transparent 75%, #2a2a2a 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
}
.poly-preview {
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
}
.poly-preview img {
    max-width: 100%;
    max-height: 100%;
}
.poly-preview span {
    color: #666;
    font-size: 12px;
}

/* Action Area */
.action-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.rotate-group {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 10px;
}
.tool-btn {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #fff !important;
    font-size: 14px !important;
    transition: all 0.3s !important;
}
.tool-btn:hover {
    background: #818cf8 !important;
    border-color: #818cf8 !important;
    transform: scale(1.05);
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
    color: rgba(255,255,255,0.6) !important;
}
.reset-btn:hover {
    color: #ff6b6b !important;
}
</style>
