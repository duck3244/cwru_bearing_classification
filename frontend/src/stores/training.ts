import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { EpochOut, TrainParams } from '@/api/client'
import { api } from '@/api/client'

type Status = 'IDLE' | 'RUNNING' | 'COMPLETED' | 'CANCELLED' | 'FAILED'

export const useTrainingStore = defineStore('training', () => {
  const status = ref<Status>('IDLE')
  const runId = ref<string | null>(null)
  const history = ref<EpochOut[]>([])
  const lastError = ref<string | null>(null)
  const lastComplete = ref<{ best_val_acc: number; promoted: boolean } | null>(null)
  let es: EventSource | null = null

  function connect() {
    es?.close()
    es = new EventSource('/api/train/events')
    es.addEventListener('snapshot', (e) => {
      const s = JSON.parse((e as MessageEvent).data)
      runId.value = s.run_id ?? null
      status.value = (s.status as Status) ?? 'IDLE'
      history.value = (s.history ?? []) as EpochOut[]
    })
    es.addEventListener('epoch', (e) => {
      const m = JSON.parse((e as MessageEvent).data) as EpochOut & { run_id: string }
      runId.value = m.run_id
      status.value = 'RUNNING'
      history.value.push(m)
    })
    es.addEventListener('complete', (e) => {
      const d = JSON.parse((e as MessageEvent).data)
      lastComplete.value = { best_val_acc: d.best_val_acc, promoted: d.promoted }
      status.value = 'COMPLETED'
    })
    es.addEventListener('cancelled', () => { status.value = 'CANCELLED' })
    es.addEventListener('error', (e) => {
      const ev = e as MessageEvent
      if (ev.data) lastError.value = JSON.parse(ev.data).message ?? 'unknown error'
    })
  }

  function disconnect() { es?.close(); es = null }

  async function start(params: TrainParams) {
    lastError.value = null; lastComplete.value = null
    history.value = []
    const { data } = await api.post<{ run_id: string }>('/train/start', params)
    runId.value = data.run_id
    status.value = 'RUNNING'
    if (!es) connect()
    return data.run_id
  }

  async function cancel() {
    await api.post('/train/cancel')
  }

  return { status, runId, history, lastError, lastComplete, connect, disconnect, start, cancel }
})
