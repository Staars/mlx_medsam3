const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  session_id: string;
  width: number;
  height: number;
  total_slices: number;
  current_slice: number;
  is_volume: boolean;
  is_dicom: boolean;
  modality: string;
  message: string;
  processing_time_ms: number;
  peak_memory_mb: number;
}

export interface RLEMask {
  counts: number[];  // Run-length encoded counts
  size: [number, number];  // [height, width]
}

export interface SegmentationResult {
  original_width: number;
  original_height: number;
  masks?: RLEMask[];      // RLE-encoded masks
  boxes?: number[][];     // [N, 4] as [x0, y0, x1, y1]
  scores?: number[];      // [N]
  prompted_boxes?: { box: number[]; label: boolean }[];
  prompted_points?: { point: number[]; label: boolean }[];
}

export interface SegmentResponse {
  session_id: string;
  prompt?: string;
  box_type?: string;
  point_type?: string;
  slice_index?: number;
  total_slices?: number;
  width?: number;
  height?: number;
  results: SegmentationResult;
  processing_time_ms: number;
  peak_memory_mb?: number;
}

export interface ResetResponse {
  session_id: string;
  message: string;
  results: SegmentationResult;
  processing_time_ms: number;
}

// Helper to fetch with error handling
async function apiFetch<T>(fetchFn: () => Promise<Response>): Promise<T> {
  const response = await fetchFn();
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Request failed");
  }
  
  return response.json();
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("files", file);

  return apiFetch<UploadResponse>(() =>
    fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    })
  );
}

export async function segmentWithText(
  sessionId: string,
  prompt: string
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/segment/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, prompt }),
    })
  );
}

export async function addBoxPrompt(
  sessionId: string,
  box: number[],
  label: boolean
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/segment/box`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, box, label }),
    })
  );
}

export async function addPointPrompt(
  sessionId: string,
  point: number[],
  label: boolean
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/segment/point`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, point, label }),
    })
  );
}

export async function resetPrompts(sessionId: string): Promise<ResetResponse> {
  return apiFetch<ResetResponse>(() =>
    fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    })
  );
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return apiFetch<{ status: string; model_loaded: boolean }>(() =>
    fetch(`${API_BASE}/health`)
  );
}

// Medical imaging APIs
export interface ModalityConfig {
  confidence_threshold: number;
  nms_threshold: number;
  [key: string]: any;
}

export interface ModalitiesResponse {
  modalities: string[];
  configs: Record<string, ModalityConfig>;
}

export interface SuggestionsResponse {
  modality: string;
  suggestions: string[];
  config: ModalityConfig;
}

export interface ModalityResponse {
  session_id: string;
  modality: string;
  config: ModalityConfig;
  suggestions: string[];
  message: string;
}

export async function getModalities(): Promise<ModalitiesResponse> {
  return apiFetch<ModalitiesResponse>(() =>
    fetch(`${API_BASE}/modalities`)
  );
}

export async function getMedicalSuggestions(modality: string): Promise<SuggestionsResponse> {
  return apiFetch<SuggestionsResponse>(() =>
    fetch(`${API_BASE}/medical/suggestions/${modality}`)
  );
}

export async function setModality(
  sessionId: string,
  modality: string
): Promise<ModalityResponse> {
  return apiFetch<ModalityResponse>(() =>
    fetch(`${API_BASE}/modality`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, modality }),
    })
  );
}

// Volume/Slice APIs
export interface VolumeInfoResponse {
  session_id: string;
  total_slices: number;
  current_slice: number;
  is_dicom: boolean;
  is_volume: boolean;
  modality: string;
  image_size: [number, number];
}

export async function changeSlice(
  sessionId: string,
  sliceIndex: number
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/slice`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, slice_index: sliceIndex }),
    })
  );
}

export async function getVolumeInfo(sessionId: string): Promise<VolumeInfoResponse> {
  return apiFetch<VolumeInfoResponse>(() =>
    fetch(`${API_BASE}/volume-info/${sessionId}`)
  );
}

// 3D Propagation APIs
export interface PropagateResponse {
  session_id: string;
  source_slice: number;
  propagated_slices: number[];
  total_propagated: number;
  direction: string;
  processing_time_ms: number;
  peak_memory_mb: number;
}

export async function propagateMasks(
  sessionId: string,
  direction: string = "both"
): Promise<PropagateResponse> {
  return apiFetch<PropagateResponse>(() =>
    fetch(`${API_BASE}/propagate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, direction }),
    })
  );
}

export async function getVolumeMask(
  sessionId: string,
  sliceIndex: number
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/volume-masks/${sessionId}/${sliceIndex}`)
  );
}
