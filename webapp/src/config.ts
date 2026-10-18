export type SignMode = 'asl' | 'isl';

export interface ModeConfig {
  label: string;
  modelUrl: string;
  labelsUrl: string;
  numHands: number;
  inputSize: number;
  channels: 1 | 3;
  /** Padding fraction applied to each side of the hand bounding box before squaring/cropping. */
  bboxPadding: number;
}

export const MODE_CONFIGS: Record<SignMode, ModeConfig> = {
  asl: {
    label: 'ASL (American)',
    modelUrl: '/tfjs_model_asl/model.json',
    labelsUrl: '/tfjs_model_asl/labels.json',
    numHands: 1,
    // Retrained model: MobileNetV2 (ImageNet transfer learning) + spatial
    // attention head, 96x96 RGB, trained on hand crops from a real, diverse
    // (varied background/lighting/skin tone) photo dataset -- replacing the
    // original Sign Language MNIST model, which was 28x28 grayscale and
    // performed poorly on live webcam input (see README.md for the full
    // story: crop-padding and letter-repeat bugs were fixed first, but real
    // accuracy only came from swapping in this dataset + architecture).
    inputSize: 96,
    channels: 3,
    // Matches the padding crop_asl2_hands.py used to build the training set,
    // so live inference sees the same framing the model was trained on.
    bboxPadding: 0.12,
  },
  isl: {
    label: 'ISL (Indian)',
    modelUrl: '/tfjs_model_isl/model.json',
    labelsUrl: '/tfjs_model_isl/labels.json',
    numHands: 2,
    inputSize: 64,
    channels: 3,
    // The ISL training photos (isl_data/sample_*.png) leave more headroom
    // around the two hands than ASL's dataset does, so less aggressive of a
    // cut is needed here -- but still tighter than the original 35%.
    bboxPadding: 0.2,
  },
};

// Only accept a prediction when the model is at least this confident.
export const CONFIDENCE_THRESHOLD = 0.6;
// Number of recent frames to majority-vote over before "locking in" a letter.
export const SMOOTHING_WINDOW = 8;
// Minimum quorum within the smoothing window required to lock in a letter.
export const SMOOTHING_MIN_AGREEMENT = 5;
// Don't re-speak the same letter more often than this (ms).
export const SPEAK_COOLDOWN_MS = 2000;
