<script setup lang="ts">

</script>

<template>
  <nav class="h-screen flex justify-center items-center">
    <ul class="items-start space-y-1 text-surface-700">
        <li>
            <div class="text-=lg font-bold pb-2">
                LOGO
            </div>
        </li>
        <li>    
            <RouterLink class="text-xl" to="/">LungVision</RouterLink>
        </li>   
        <li>
            <RouterLink class="text-xl" to="/demo/">Demo</RouterLink>
        </li>
        <li>
            <RouterLink class="text-xl" to="/research/">Research</RouterLink>
        </li>
        <li>
            <RouterLink class="text-xl" to="/team/">Team</RouterLink>
        </li>
    </ul>
  </nav>
</template>

<style lang="scss" scoped>

</style>