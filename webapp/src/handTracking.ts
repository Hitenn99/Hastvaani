import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';

// Loaded from a CDN (jsdelivr for the WASM runtime, Google Cloud Storage for
// the model weights) rather than bundled -- this keeps the app a pure static
// site with no backend, at the cost of needing network access on first load
// (the browser then caches both).
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker: HandLandmarker | null = null;
let currentNumHands = -1;
let currentMinDetection = -1;

/** Lazily creates (or reconfigures, if numHands / the detection threshold changed) the shared
 * HandLandmarker instance. The Sentence Builder passes 0.3, the threshold its reference banks
 * were extracted with (model_assets/sentences/extract_landmarks.py), so live detection matches. */
export async function getHandLandmarker(numHands: number, minDetection = 0.5): Promise<HandLandmarker> {
  if (landmarker && currentNumHands === numHands && currentMinDetection === minDetection) {
    return landmarker;
  }
  if (landmarker) {
    landmarker.close();
    landmarker = null;
  }
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands,
    minHandDetectionConfidence: minDetection,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  currentNumHands = numHands;
  currentMinDetection = minDetection;
  return landmarker;
}

export function detectHands(video: HTMLVideoElement, timestampMs: number): HandLandmarkerResult | null {
  if (!landmarker) return null;
  return landmarker.detectForVideo(video, timestampMs);
}

export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Computes a padded, square, center-preserving bounding box (in pixel
 * coordinates) around one or more hands' landmarks. For ISL (two hands) this
 * is the union box of all detected hands, per the spec.
 */
export function computeUnionBBox(
  result: HandLandmarkerResult,
  videoWidth: number,
  videoHeight: number,
  paddingFraction: number,
): BBox | null {
  if (!result.landmarks || result.landmarks.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const handLandmarks of result.landmarks) {
    for (const lm of handLandmarks) {
      const px = lm.x * videoWidth;
      const py = lm.y * videoHeight;
      if (px < minX) minX = px;
      if (py < minY) minY = py;
      if (px > maxX) maxX = px;
      if (py > maxY) maxY = py;
    }
  }

  const boxWidth = maxX - minX;
  const boxHeight = maxY - minY;
  const padX = boxWidth * paddingFraction;
  const padY = boxHeight * paddingFraction;

  minX -= padX;
  maxX += padX;
  minY -= padY;
  maxY += padY;

  // Square the box around its center so the crop isn't stretched on resize.
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const side = Math.max(maxX - minX, maxY - minY);

  let x = cx - side / 2;
  let y = cy - side / 2;

  // Clamp to the video frame bounds.
  x = Math.max(0, Math.min(x, videoWidth - 1));
  y = Math.max(0, Math.min(y, videoHeight - 1));
  const width = Math.min(side, videoWidth - x);
  const height = Math.min(side, videoHeight - y);

  return { x, y, width, height };
}
