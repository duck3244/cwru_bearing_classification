<script setup lang="ts">
import { ref } from 'vue'
import axios from 'axios'
import { api, type InferenceResponse } from '@/api/client'

const file = ref<File | null>(null)
const result = ref<InferenceResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

function onFile(e: Event) {
  const input = e.target as HTMLInputElement
  file.value = input.files?.[0] ?? null
}

async function submit() {
  if (!file.value) return
  loading.value = true; error.value = null; result.value = null
  try {
    const fd = new FormData()
    fd.append('file', file.value)
    const { data } = await api.post<InferenceResponse>('/predict', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    result.value = data
  } catch (e) {
    if (axios.isAxiosError(e)) error.value = (e.response?.data as { detail?: string })?.detail ?? e.message
    else error.value = (e as Error).message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="space-y-4">
    <h2 class="text-2xl font-semibold">Predict</h2>
    <div class="bg-white border rounded-lg p-4 shadow-sm space-y-3">
      <input type="file" accept=".mat" @change="onFile" class="block" />
      <button
        @click="submit"
        :disabled="!file || loading"
        class="px-4 py-2 rounded bg-slate-900 text-white disabled:bg-slate-300"
      >{{ loading ? 'Predicting...' : 'Predict' }}</button>
      <p v-if="error" class="text-rose-600 text-sm">{{ error }}</p>
    </div>

    <div v-if="result" class="bg-white border rounded-lg p-4 shadow-sm space-y-2">
      <div class="flex items-baseline gap-3">
        <h3 class="text-lg font-semibold">{{ result.aggregated.label_name }}</h3>
        <span class="text-sm text-slate-500">{{ result.n_windows }} windows · {{ result.processing_ms }}ms</span>
      </div>
      <div class="grid grid-cols-4 gap-2 text-sm">
        <div v-for="(p, name) in result.aggregated.probabilities" :key="name"
             class="border rounded p-2 text-center"
             :class="name === result.aggregated.label_name ? 'border-emerald-500 bg-emerald-50' : ''">
          <div class="font-medium">{{ name }}</div>
          <div class="text-slate-500">{{ (p * 100).toFixed(2) }}%</div>
        </div>
      </div>
      <details class="mt-2">
        <summary class="cursor-pointer text-sm text-slate-600">Window-level (first 20)</summary>
        <table class="text-xs w-full mt-2">
          <thead><tr class="text-left border-b">
            <th class="py-1">idx</th><th>start</th><th>label</th>
          </tr></thead>
          <tbody>
            <tr v-for="w in result.windows.slice(0, 20)" :key="w.idx" class="border-b">
              <td class="py-0.5">{{ w.idx }}</td>
              <td>{{ w.start }}</td>
              <td>{{ w.label_name }}</td>
            </tr>
          </tbody>
        </table>
      </details>
    </div>
  </section>
</template>
