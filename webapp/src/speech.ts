import { SPEAK_COOLDOWN_MS } from './config';

export class SpeechAnnouncer {
  private muted = false;
  private lastSpoken: string | null = null;
  private lastSpokenAt = 0;
  private supported = typeof window !== 'undefined' && 'speechSynthesis' in window;

  isSupported() {
    return this.supported;
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    if (muted && this.supported) {
      window.speechSynthesis.cancel();
    }
  }

  isMuted() {
    return this.muted;
  }

  /** Speaks `letter` unless muted, unsupported, or spoken too recently. */
  announce(letter: string) {
    if (this.muted || !this.supported) return;
    const now = Date.now();
    if (letter === this.lastSpoken && now - this.lastSpokenAt < SPEAK_COOLDOWN_MS) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(letter);
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
    this.lastSpoken = letter;
    this.lastSpokenAt = now;
  }

  speakText(text: string) {
    if (this.muted || !this.supported || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  }
}
