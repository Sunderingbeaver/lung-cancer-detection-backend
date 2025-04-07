<template>
  <div class="flex h-screen">
    <div class="w-[30%] bg-rose-200">
      <PoweredByJethan />
    </div>
    <div class="w-[70%] bg-red-100 overflow-auto">
      <router-view v-slot="{ Component }">
        <transition name="fade">
          <component :is="Component" />
        </transition>
      </router-view>
    </div>
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>