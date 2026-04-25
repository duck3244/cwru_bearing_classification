<script setup lang="ts">
import {
  CategoryScale, Chart as ChartJS, Legend, LinearScale, LineElement,
  PointElement, Title, Tooltip,
} from 'chart.js'
import { Line } from 'vue-chartjs'
import { computed } from 'vue'
import type { EpochOut } from '@/api/client'

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Title, Tooltip, Legend)

const props = defineProps<{ history: EpochOut[]; mode: 'loss' | 'acc' }>()

const chartData = computed(() => {
  const labels = props.history.map(h => String(h.epoch))
  if (props.mode === 'loss') {
    return {
      labels,
      datasets: [
        { label: 'Train Loss', data: props.history.map(h => h.train_loss), borderColor: '#0ea5e9', tension: 0.2, fill: false },
        { label: 'Val Loss', data: props.history.map(h => h.val_loss), borderColor: '#f43f5e', tension: 0.2, fill: false },
      ],
    }
  }
  return {
    labels,
    datasets: [
      { label: 'Train Acc', data: props.history.map(h => h.train_acc), borderColor: '#0ea5e9', tension: 0.2, fill: false },
      { label: 'Val Acc', data: props.history.map(h => h.val_acc), borderColor: '#10b981', tension: 0.2, fill: false },
    ],
  }
})
const options = { responsive: true, maintainAspectRatio: false, animation: false as const }
</script>

<template>
  <div class="h-72">
    <Line :data="chartData" :options="options" />
  </div>
</template>
