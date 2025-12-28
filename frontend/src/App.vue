<template>
  <div class="layout-container">
    <AppSidebar />
    <div class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
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
  /* Global Background */
  background: radial-gradient(at 0% 0%, #2b3595 0px, transparent 50%),
              radial-gradient(at 98% 1%, #7928ca 0px, transparent 50%),
              radial-gradient(at 4% 97%, #ff0f7b 0px, transparent 50%),
              #1a1a2e;
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
</style>
