//PrimeVue
import PrimeVue from 'primevue/config';
import { preset } from './assets/primevuepreset';
import ToastService from 'primevue/toastservice';

// Styles
import '@fontsource-variable/dm-sans'
import './assets/main.css'

// Extensions
import { createPinia } from 'pinia'
import { router } from './router'

// Vue Core
import { createApp } from 'vue'
import App from './App.vue'
import { option } from '@primeuix/themes/aura/autocomplete';
const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(PrimeVue, {
    theme: {
        preset,
        options: {
            cssLayer: {
                name: "primevue",
                order: "tailwind-base, primevue, tailwind-utilities",
            },
            darkModeSelector: ".dark-mode",
        },
    }
})
app.use(ToastService)

app.mount('#app')
