<template>
  <div class="layout-container">
    <AppSidebar />
    <div class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
      
      <div class="main-footer">
          Powered by DerainSys &copy; 2025 | 合肥工业大学梁禹设计
      </div>
    </div>
  </div>
</template>

<script setup>
import AppSidebar from './components/AppSidebar.vue'
</script>

<style>
body {
  margin: 0;
  padding: 0;
  background-color: #1a1a2e;
}
.layout-container {
  display: flex;
  min-height: 100vh;
  /* Dynamic Background */
  background: linear-gradient(-45deg, #1a1a2e, #2b3595, #7928ca, #ff0f7b);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  height: 100vh;
}

/* Page Transitions: Slide Up + Fade */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.4s cubic-bezier(0.25, 1, 0.5, 1);
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(20px) scale(0.98);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-20px) scale(0.98);
}

#app {
  font-family: 'Inter', Helvetica, Arial, sans-serif;
  color: #fff;
}

.main-footer {
  text-align: center;
  padding: 20px;
  color: rgba(255, 255, 255, 0.3);
  font-size: 12px;
  margin-top: auto;
}
</style>
