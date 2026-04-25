<script setup lang="ts">
import { onMounted, onUnmounted, reactive } from 'vue'
import { storeToRefs } from 'pinia'
import { useTrainingStore } from '@/stores/training'
import MetricChart from '@/components/MetricChart.vue'

const store = useTrainingStore()
const { status, history, runId, lastError, lastComplete } = storeToRefs(store)

const params = reactive({
  EPOCHS: 50,
  BATCH_SIZE: 32,
  LEARNING_RATE: 0.001,
  WEIGHT_DECAY: 0.00001,
  DROPOUT: 0.3,
  EARLY_STOPPING_PATIENCE: 10,
})

async function start() { await store.start(params) }
async function cancel() { await store.cancel() }

onMounted(() => store.connect())
onUnmounted(() => store.disconnect())
</script>

<template>
  <section class="space-y-4">
    <h2 class="text-2xl font-semibold">Train</h2>

    <div class="bg-white border rounded-lg p-4 shadow-sm grid grid-cols-2 md:grid-cols-3 gap-3">
      <label class="text-sm">EPOCHS<input v-model.number="params.EPOCHS" type="number" min="1" class="w-full border rounded px-2 py-1"/></label>
      <label class="text-sm">BATCH_SIZE<input v-model.number="params.BATCH_SIZE" type="number" min="1" class="w-full border rounded px-2 py-1"/></label>
      <label class="text-sm">LEARNING_RATE<input v-model.number="params.LEARNING_RATE" type="number" step="0.0001" class="w-full border rounded px-2 py-1"/></label>
      <label class="text-sm">WEIGHT_DECAY<input v-model.number="params.WEIGHT_DECAY" type="number" step="0.00001" class="w-full border rounded px-2 py-1"/></label>
      <label class="text-sm">DROPOUT<input v-model.number="params.DROPOUT" type="number" step="0.05" min="0" max="0.9" class="w-full border rounded px-2 py-1"/></label>
      <label class="text-sm">EARLY_STOPPING<input v-model.number="params.EARLY_STOPPING_PATIENCE" type="number" min="1" class="w-full border rounded px-2 py-1"/></label>
    </div>

    <div class="flex gap-2">
      <button @click="start" :disabled="status === 'RUNNING'"
              class="px-4 py-2 rounded bg-slate-900 text-white disabled:bg-slate-300">Start</button>
      <button @click="cancel" :disabled="status !== 'RUNNING'"
              class="px-4 py-2 rounded border border-rose-500 text-rose-600 disabled:opacity-40">Cancel</button>
      <span class="text-sm text-slate-600 ml-3">
        status: <b>{{ status }}</b><span v-if="runId"> · run {{ runId.slice(0,8) }}</span>
      </span>
    </div>

    <p v-if="lastError" class="text-rose-600 text-sm">{{ lastError }}</p>
    <p v-if="lastComplete" class="text-emerald-700 text-sm">
      Completed · best_val_acc={{ lastComplete.best_val_acc.toFixed(2) }}%
      · {{ lastComplete.promoted ? 'promoted' : 'not promoted' }}
    </p>

    <div v-if="history.length" class="grid md:grid-cols-2 gap-4">
      <div class="bg-white border rounded-lg p-3 shadow-sm">
        <h3 class="text-sm font-medium mb-2">Loss</h3>
        <MetricChart :history="history" mode="loss" />
      </div>
      <div class="bg-white border rounded-lg p-3 shadow-sm">
        <h3 class="text-sm font-medium mb-2">Accuracy</h3>
        <MetricChart :history="history" mode="acc" />
      </div>
    </div>
  </section>
</template>
