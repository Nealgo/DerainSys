import { createRouter, createWebHistory } from 'vue-router'
import ImageRestore from '../components/ImageRestore.vue'
import ImageStitch from '../views/ImageStitch.vue'
import ImageCrop from '../views/ImageCrop.vue'
import ImageAdjust from '../views/ImageAdjust.vue'

const routes = [
  {
    path: '/',
    name: 'Derain',
    component: ImageRestore
  },
  {
    path: '/stitch',
    name: 'Stitch',
    component: ImageStitch
  },
  {
    path: '/crop',
    name: 'Crop',
    component: ImageCrop
  },
  {
    path: '/color',
    name: 'Color',
    component: ImageAdjust
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
