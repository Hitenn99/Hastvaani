// Digit sign descriptions + a small finger-state diagram used on the Math page.
// The diagrams are drawn from the same finger rules the classifier uses (digits.ts), so what the
// page shows is what the recogniser checks for.

export interface DigitGuide {
  digit: number;
  ext: [boolean, boolean, boolean, boolean]; // index, middle, ring, little extended
  thumb: 'out' | 'in' | 0 | 1 | 2 | 3; // out / tucked / touching a fingertip (0 index .. 3 little)
  text: string;
}

export const GUIDE: DigitGuide[] = [
  { digit: 0, ext: [false, false, false, false], thumb: 0, text: 'Curl all fingers and bring the thumb tip to meet them, making an O.' },
  { digit: 1, ext: [true, false, false, false], thumb: 'in', text: 'Index finger straight up. Other fingers and thumb folded.' },
  { digit: 2, ext: [true, true, false, false], thumb: 'in', text: 'Index and middle fingers up in a V. Ring and little finger folded.' },
  { digit: 3, ext: [true, true, false, false], thumb: 'out', text: 'Thumb, index and middle finger out. Ring and little finger folded.' },
  { digit: 4, ext: [true, true, true, true], thumb: 'in', text: 'Four fingers up, thumb tucked into the palm.' },
  { digit: 5, ext: [true, true, true, true], thumb: 'out', text: 'Open hand, all five fingers spread with the thumb out.' },
  { digit: 6, ext: [true, true, true, false], thumb: 3, text: 'Index, middle and ring finger up. Thumb touches the little finger.' },
  { digit: 7, ext: [true, true, false, true], thumb: 2, text: 'Index, middle and little finger up. Thumb touches the ring finger.' },
  { digit: 8, ext: [true, false, true, true], thumb: 1, text: 'Index, ring and little finger up. Thumb touches the middle finger.' },
  { digit: 9, ext: [false, true, true, true], thumb: 0, text: 'Middle, ring and little finger up. Thumb touches the index finger.' },
];

const X = [30, 43, 56, 69]; // finger x centres
const UP_TOP = [14, 6, 12, 24]; // top y of an extended finger (index, middle, ring, little)
const FOLD_TOP = 50;

/** Returns an inline SVG string (hand seen palm-side-out, fingers up). */
export function fingerDiagram(g: DigitGuide): string {
  const fingers = g.ext
    .map((up, i) => {
      const top = up ? UP_TOP[i] : FOLD_TOP;
      const h = 62 - top;
      const fill = up ? 'var(--color-brand)' : 'color-mix(in srgb, var(--color-ink) 22%, transparent)';
      return `<rect x="${X[i] - 5.5}" y="${top}" width="11" height="${h}" rx="5.5" fill="${fill}"/>`;
    })
    .join('');

  let thumb = '';
  if (g.thumb === 'out') {
    thumb = `<rect x="3" y="52" width="30" height="11" rx="5.5" fill="var(--color-accent)" transform="rotate(-38 30 62)"/>`;
  } else if (g.thumb === 'in') {
    thumb = `<rect x="26" y="66" width="26" height="10" rx="5" fill="var(--color-accent)" opacity=".85"/>`;
  } else {
    const tx = X[g.thumb];
    const tipY = g.ext[g.thumb] ? UP_TOP[g.thumb] + 6 : FOLD_TOP + 6;
    thumb = `<path d="M22 84 Q ${tx - 14} 64 ${tx} ${tipY}" stroke="var(--color-accent)" stroke-width="10" stroke-linecap="round" fill="none"/>
             <circle cx="${tx}" cy="${tipY}" r="8" fill="none" stroke="var(--color-accent)" stroke-width="2.5"/>`;
  }

  return `<svg viewBox="0 0 100 110" class="mx-auto h-28 w-auto" role="img" aria-hidden="true">
    ${fingers}
    <rect x="22" y="56" width="56" height="42" rx="16" fill="color-mix(in srgb, var(--color-ink) 12%, var(--color-surface))" stroke="color-mix(in srgb, var(--color-ink) 25%, transparent)" stroke-width="1.5"/>
    ${thumb}
  </svg>`;
}

// ---------- Per-language guides (each built from its own chart image) ----------
export type Lang = 'asl' | 'isl';

export interface GuideCard {
  digit: number;
  img?: string; // cropped from that language's chart
  text: string;
  note?: string;
  referenceOnly?: boolean; // shown for completeness, not recognised
}

const T_ASL: Record<number, string> = {
  0: 'Curl all fingers and bring the thumb tip to meet them, making an O.',
  1: 'Index finger straight up; the other fingers are folded over the thumb.',
  2: 'Index and middle fingers up in a V; ring and little finger folded and held by the thumb.',
  3: 'Thumb, index and middle finger stretched out; ring and little finger folded.',
  4: 'Four fingers up with the thumb tucked across the palm.',
  5: 'Open hand: all five fingers spread and the thumb out.',
  6: 'Index, middle and ring finger up; the thumb touches the little finger.',
  7: 'Thumb touches the ring finger; index, middle and little finger up.',
  8: 'Thumb touches the middle finger; index, ring and little finger up.',
  9: 'Thumb touches the index finger; middle, ring and little finger up.',
};

const T_ISL: Record<number, string> = {
  0: 'Fingers curved together and meeting the thumb, making a rounded O / C shape.',
  1: 'Index finger straight up; the other fingers curled into the palm.',
  2: 'Index and middle fingers up in a V; the other fingers folded.',
  3: 'Index and middle fingers up with the thumb stretched out to the side; ring and little finger folded.',
  4: 'Four fingers up with the thumb tucked into the palm.',
  5: 'Open hand with all five fingers spread and the thumb stretched sideways.',
  6: 'Index, middle and ring finger up; the little finger is folded across the palm and held down by the thumb.',
  7: 'Thumb touches the ring finger; index, middle and little finger up.',
  8: 'Thumb touches the middle finger; index, ring and little finger up.',
  9: 'Thumb touches the index finger; the other three fingers up.',
};

export const GUIDE_BY_LANG: Record<Lang, GuideCard[]> = {
  asl: [
    { digit: 0, text: T_ASL[0], note: 'Not in the supplied ASL chart; this is the standard ASL zero (same as the letter O).' },
    ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => ({ digit: d, img: `/signs/num_asl/${d}.png`, text: T_ASL[d] })),
  ],
  isl: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => ({ digit: d, img: `/signs/num_isl/${d}.png`, text: T_ISL[d] })),
};

export const GUIDE_NOTES: Record<Lang, string> = {
  asl:
    'ASL number chart. Digits 1–9 are shown as drawn in the chart. ' +
    'Chart supplied by the site owner.',
  isl:
    'ISL numeral chart, 0–9 (Ekbote & Joshi, “Indian sign language recognition using ANN and SVM classifiers”, 2017, Fig. 1). ' +
    'ISL has its own 0 (a curled hand), and its 6 folds the little finger across the palm under the thumb.',
};

export const SHARED_NOTE =
  'In these two charts digits 1–5 and 7–9 use the same handshapes, so one recogniser handles both languages.';
