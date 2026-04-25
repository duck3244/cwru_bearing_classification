import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/predict' },
    { path: '/predict', component: () => import('@/views/PredictView.vue') },
    { path: '/train', component: () => import('@/views/TrainView.vue') },
    { path: '/history', component: () => import('@/views/HistoryView.vue') },
  ],
})

export default router
