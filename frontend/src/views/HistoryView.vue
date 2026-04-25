<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api, type ArtifactOut, type InferenceLogOut, type RunOut } from '@/api/client'

const runs = ref<RunOut[]>([])
const artifacts = ref<ArtifactOut[]>([])
const inferences = ref<InferenceLogOut[]>([])
const loading = ref(false)

async function refresh() {
  loading.value = true
  try {
    const [r, m, i] = await Promise.all([
      api.get<RunOut[]>('/train/runs'),
      api.get<ArtifactOut[]>('/models'),
      api.get<InferenceLogOut[]>('/inferences'),
    ])
    runs.value = r.data; artifacts.value = m.data; inferences.value = i.data
  } finally {
    loading.value = false
  }
}

async function promote(id: string) {
  await api.post(`/models/${id}/promote`)
  await refresh()
}

onMounted(refresh)
</script>

<template>
  <section class="space-y-6">
    <div class="flex items-center justify-between">
      <h2 class="text-2xl font-semibold">History</h2>
      <button @click="refresh" :disabled="loading"
              class="px-3 py-1.5 text-sm rounded border">Refresh</button>
    </div>

    <div>
      <h3 class="text-lg font-medium mb-2">Training Runs</h3>
      <div class="bg-white border rounded-lg shadow-sm overflow-x-auto">
        <table class="text-sm w-full">
          <thead class="bg-slate-100 text-left">
            <tr><th class="px-3 py-2">id</th><th>started</th><th>status</th>
                <th>best_val_acc</th><th>epochs</th><th>artifact</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in runs" :key="r.id" class="border-t">
              <td class="px-3 py-1.5 font-mono text-xs">{{ r.id.slice(0,8) }}</td>
              <td class="text-xs">{{ r.started_at }}</td>
              <td><span :class="{
                'text-emerald-600': r.status === 'COMPLETED',
                'text-rose-600': r.status === 'FAILED',
                'text-amber-600': r.status === 'RUNNING' || r.status === 'CANCELLED',
              }">{{ r.status }}</span></td>
              <td>{{ r.best_val_acc?.toFixed(2) ?? '—' }}</td>
              <td>{{ r.n_epochs_run }}</td>
              <td class="font-mono text-xs">{{ r.artifact_id?.slice(0,8) ?? '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-medium mb-2">Model Artifacts</h3>
      <div class="bg-white border rounded-lg shadow-sm overflow-x-auto">
        <table class="text-sm w-full">
          <thead class="bg-slate-100 text-left">
            <tr><th class="px-3 py-2">id</th><th>val_acc</th><th>created</th>
                <th>current</th><th></th></tr>
          </thead>
          <tbody>
            <tr v-for="a in artifacts" :key="a.id" class="border-t">
              <td class="px-3 py-1.5 font-mono text-xs">{{ a.id.slice(0,8) }}</td>
              <td>{{ a.val_acc.toFixed(2) }}</td>
              <td class="text-xs">{{ a.created_at }}</td>
              <td><span v-if="a.is_current" class="text-emerald-600 font-medium">current</span></td>
              <td>
                <button v-if="!a.is_current" @click="promote(a.id)"
                        class="px-2 py-0.5 text-xs border rounded">Promote</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div>
      <h3 class="text-lg font-medium mb-2">Inference Logs</h3>
      <div class="bg-white border rounded-lg shadow-sm overflow-x-auto">
        <table class="text-sm w-full">
          <thead class="bg-slate-100 text-left">
            <tr><th class="px-3 py-2">time</th><th>file</th><th>label</th>
                <th>windows</th><th>ms</th></tr>
          </thead>
          <tbody>
            <tr v-for="i in inferences" :key="i.id" class="border-t">
              <td class="px-3 py-1.5 text-xs">{{ i.timestamp }}</td>
              <td class="text-xs">{{ i.filename }}</td>
              <td>{{ i.aggregated_label_name }}</td>
              <td>{{ i.n_windows }}</td>
              <td>{{ i.processing_ms }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>
</template>
