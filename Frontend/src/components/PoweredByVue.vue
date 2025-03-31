<script setup lang="ts">
import vue from '@/assets/vue.png'
</script>

<template>
  <div class="flex justify-end p-4">
    <div class="text-muted-color">Powered by</div>
    <div class="flex items-center -ml-12 pt-6 text-green-400 font-semibold">
      <img
        :src="vue"
        width="50px"
      />
      <span>Vue JS</span>
    </div>
  </div>
</template>
