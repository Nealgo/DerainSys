<template>
  <div class="w-72 h-full flex flex-col bg-white/60 backdrop-blur-xl border-r border-white/50 shadow-[4px_0_24px_rgba(0,0,0,0.02)] transition-all duration-300">
    <!-- Branding -->
    <div class="h-24 flex items-center justify-center border-b border-slate-100/50">
      <h1 class="text-2xl font-black bg-gradient-to-br from-blue-600 to-violet-600 bg-clip-text text-transparent cursor-default select-none tracking-tight hover:scale-105 transition-transform duration-300">
        DerainSys
      </h1>
    </div>
    
    <!-- Navigation -->
    <nav class="flex-1 py-8 px-4 space-y-3">
      <router-link 
        v-for="item in menuItems" 
        :key="item.path" 
        :to="item.path"
        class="group flex items-center px-5 py-4 rounded-2xl transition-all duration-300 relative overflow-hidden"
        :class="isActive(item.path) ? 'bg-blue-600 shadow-blue-300/50 shadow-lg translate-x-1' : 'hover:bg-white hover:shadow-md hover:translate-x-1 hover:text-blue-600 text-slate-500'"
      >
        <!-- Active Background Glow -->
        <div v-if="isActive(item.path)" class="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 opacity-100 z-0"></div>

        <!-- Icon -->
        <el-icon 
          class="text-xl mr-4 relative z-10 transition-transform duration-300 group-hover:scale-110 group-hover:rotate-6"
          :class="isActive(item.path) ? 'text-white' : 'text-slate-400 group-hover:text-blue-500'"
        >
          <component :is="item.icon" />
        </el-icon>
        
        <!-- Text -->
        <span 
          class="font-semibold tracking-wide relative z-10"
          :class="isActive(item.path) ? 'text-white' : ''"
        >
          {{ item.name }}
        </span>

        <!-- Hover Indicator (Right side pill) -->
        <div v-if="!isActive(item.path)" class="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-blue-500 rounded-l-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
      </router-link>
    </nav>
    
    <!-- Version -->
    <div class="p-6 text-center text-xs text-slate-300 font-medium tracking-widest uppercase">
      V 2.0 Pro
    </div>
  </div>
</template>

<script setup>
import { useRoute } from 'vue-router'
import { MagicStick, Crop, Switch, Setting } from '@element-plus/icons-vue'

const route = useRoute()

const menuItems = [
  { name: '智能去雨', path: '/', icon: MagicStick },
  { name: '图片拼接', path: '/stitch', icon: Switch },
  { name: '图片裁剪', path: '/crop', icon: Crop },
  { name: '色彩调整', path: '/adjust', icon: Setting },
]

const isActive = (path) => {
  return route.path === path
}
</script>

<style scoped>
/* No scoped CSS needed, utilizing Tailwind utilities */
</style>
