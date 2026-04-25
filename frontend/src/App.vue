<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import { onMounted, ref } from 'vue'
import { api } from '@/api/client'

const health = ref<{ device: string; model_loaded: boolean; training_running: boolean } | null>(null)

async function refresh() {
  try { health.value = (await api.get('/health')).data } catch { health.value = null }
}
onMounted(() => { refresh(); setInterval(refresh, 5000) })
</script>

<template>
  <div class="min-h-full">
    <header class="bg-slate-900 text-slate-100 px-6 py-3 flex items-center gap-6">
      <h1 class="text-lg font-semibold">CWRU Bearing Classifier</h1>
      <nav class="flex gap-4 text-sm">
        <RouterLink to="/predict" class="hover:text-cyan-300" active-class="text-cyan-300">Predict</RouterLink>
        <RouterLink to="/train" class="hover:text-cyan-300" active-class="text-cyan-300">Train</RouterLink>
        <RouterLink to="/history" class="hover:text-cyan-300" active-class="text-cyan-300">History</RouterLink>
      </nav>
      <div class="ml-auto text-xs text-slate-300 flex gap-3" v-if="health">
        <span>device: <b class="text-cyan-300">{{ health.device }}</b></span>
        <span>model: <b :class="health.model_loaded ? 'text-emerald-400' : 'text-rose-400'">{{ health.model_loaded ? 'loaded' : 'none' }}</b></span>
        <span>training: <b :class="health.training_running ? 'text-amber-400' : 'text-emerald-400'">{{ health.training_running ? 'on' : 'idle' }}</b></span>
      </div>
    </header>
    <main class="p-6 max-w-6xl mx-auto">
      <RouterView />
    </main>
  </div>
</template>
