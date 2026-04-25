import axios from 'axios'

export const api = axios.create({ baseURL: '/api', timeout: 60_000 })

// 백엔드 Pydantic 모델과 손으로 매칭. 향후 openapi-typescript로 자동화 권장.
export interface HealthOut {
  status: string
  device: string
  model_loaded: boolean
  current_artifact_id: string | null
  training_running: boolean
}

export interface CurrentModelOut {
  artifact_id: string | null
  val_acc: number | null
  val_loss: number | null
  arch_hash: string | null
  class_names: string[] | null
  input_size: number | null
  hidden_sizes: number[] | null
  created_at: string | null
  device: string
}

export interface ArtifactOut {
  id: string
  run_id: string
  val_acc: number
  val_loss: number
  is_current: boolean
  arch_hash: string
  created_at: string
}

export interface AggregatedOut {
  label: number
  label_name: string
  probabilities: Record<string, number>
}
export interface WindowOut {
  idx: number
  start: number
  label: number
  label_name: string
  probabilities: number[]
}
export interface InferenceResponse {
  filename: string
  model_artifact_id: string
  model_val_acc: number | null
  n_windows: number
  aggregated: AggregatedOut
  windows: WindowOut[]
  processing_ms: number
}
export interface InferenceLogOut {
  id: string
  timestamp: string
  filename: string
  n_windows: number
  aggregated_label: number
  aggregated_label_name: string
  aggregated_probs: Record<string, number>
  processing_ms: number
  artifact_id: string
}

export interface TrainParams {
  EPOCHS?: number
  BATCH_SIZE?: number
  LEARNING_RATE?: number
  WEIGHT_DECAY?: number
  DROPOUT?: number
  HIDDEN_SIZES?: number[]
  EARLY_STOPPING_PATIENCE?: number
}
export interface EpochOut {
  epoch: number
  train_loss: number
  train_acc: number
  val_loss: number
  val_acc: number
  lr: number
  elapsed_ms: number
}
export interface RunOut {
  id: string
  started_at: string
  finished_at: string | null
  status: string
  best_val_acc: number | null
  best_val_loss: number | null
  best_epoch: number | null
  n_epochs_run: number
  artifact_id: string | null
  error_message: string | null
}
export interface RunDetail extends RunOut {
  epochs: EpochOut[]
  config_snapshot: Record<string, unknown>
}
