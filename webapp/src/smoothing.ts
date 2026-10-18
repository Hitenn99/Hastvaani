import { SMOOTHING_WINDOW, SMOOTHING_MIN_AGREEMENT } from './config';

// How many *consecutive* low-confidence/no-hand frames it takes to actually
// drop the current lock. A single missed frame (hand jitter, motion blur, a
// blink of low confidence) should NOT immediately unlock the current letter
// -- if it did, the very next stable frame has to re-earn quorum from an
// empty buffer, which re-triggers `changed: true` and re-appends the *same*
// letter to the word buffer every time it happens. That produced the
// "HHHHHH", "GGGGGG" spam a real user hit: holding one steady sign, with
// occasional single-frame confidence dips, kept re-locking the same letter
// over and over. Requiring several consecutive misses before dropping the
// lock fixes that, while still eventually clearing it so the *same* letter
// can be intentionally re-signed later (e.g. spelling "HELLO"'s double L).
const MISS_RESET_THRESHOLD = 10;

/**
 * Majority-vote smoothing over the last N accepted (confidence-passing)
 * predictions, so a single noisy frame can't flip the displayed letter.
 * A letter is only "locked in" once it holds a quorum within the window,
 * and stays locked in through brief dips so it isn't re-announced/re-appended
 * repeatedly while the user holds one sign steady.
 */
export class PredictionSmoother {
  private buffer: string[] = [];
  private lockedLetter: string | null = null;
  private missStreak = 0;

  /** Feed one frame's prediction (already confidence-thresholded by the caller). */
  push(letter: string | null): { locked: string | null; changed: boolean } {
    if (letter === null) {
      this.missStreak++;
      if (this.missStreak < MISS_RESET_THRESHOLD) {
        // Brief dip -- keep the current lock and buffer intact.
        return { locked: this.lockedLetter, changed: false };
      }
      this.buffer = [];
      const changed = this.lockedLetter !== null;
      this.lockedLetter = null;
      return { locked: null, changed };
    }

    this.missStreak = 0;
    this.buffer.push(letter);
    if (this.buffer.length > SMOOTHING_WINDOW) {
      this.buffer.shift();
    }

    const counts = new Map<string, number>();
    for (const l of this.buffer) {
      counts.set(l, (counts.get(l) ?? 0) + 1);
    }

    let bestLetter: string | null = null;
    let bestCount = 0;
    for (const [l, c] of counts) {
      if (c > bestCount) {
        bestCount = c;
        bestLetter = l;
      }
    }

    if (bestLetter !== null && bestCount >= SMOOTHING_MIN_AGREEMENT) {
      const changed = this.lockedLetter !== bestLetter;
      this.lockedLetter = bestLetter;
      return { locked: bestLetter, changed };
    }

    return { locked: this.lockedLetter, changed: false };
  }

  reset() {
    this.buffer = [];
    this.lockedLetter = null;
    this.missStreak = 0;
  }
}
