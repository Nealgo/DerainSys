<template>
  <div class="flex h-screen w-full overflow-hidden relative bg-slate-50 font-sans text-slate-800">
    <!-- Dynamic Background Elements -->
    <div class="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div class="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] rounded-full bg-purple-200/40 blur-[120px] animate-[blob_20s_infinite]"></div>
        <div class="absolute top-[20%] -right-[10%] w-[40%] h-[40%] rounded-full bg-blue-200/40 blur-[100px] animate-[blob_25s_infinite_reverse]"></div>
        <div class="absolute -bottom-[20%] left-[20%] w-[60%] h-[60%] rounded-full bg-indigo-200/40 blur-[140px] animate-[blob_30s_infinite]"></div>
    </div>

    <!-- Sidebar (High z-index to sit above background) -->
    <AppSidebar class="z-20 relative" />

    <!-- Main Content Area -->
    <div class="flex-1 flex flex-col h-full relative z-10 transition-all duration-300">
      <router-view v-slot="{ Component }">
        <transition name="fade-slide" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
      
      <!-- Footer Info (Optional, kept subtle) -->
      <div class="text-center py-4 text-xs text-slate-400 select-none">
        Powered by DerainSys © 2025 | 合肥工业大学梁禹设计
      </div>
    </div>
  </div>
</template>

<script setup>
import AppSidebar from './components/AppSidebar.vue'
</script>

<style>
/* Global Transitions */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: opacity 0.4s ease, transform 0.4s ease;
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateY(15px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: scale(0.98);
}

/* Custom Animation Keyframes */
@keyframes blob {
  0% { transform: translate(0px, 0px) scale(1); }
  33% { transform: translate(30px, -50px) scale(1.1); }
  66% { transform: translate(-20px, 20px) scale(0.9); }
  100% { transform: translate(0px, 0px) scale(1); }
}
</style>
