// Styles
import '@fontsource-variable/dm-sans'
import './assets/main.css'

// Extensions
import { createPinia } from 'pinia'
import { router } from './router'

// Vue Core
import { createApp } from 'vue'
import App from './App.vue'
const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')
