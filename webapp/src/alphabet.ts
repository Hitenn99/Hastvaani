// Static content for the Learn page (UI only -- not used by the recognizer).

export interface LetterInfo {
  letter: string;
  hands: 1 | 2 | 'either';
  tip: string;
  motion?: boolean;
}

const asl: Record<string, string> = {
  A: 'Fist with the thumb resting along the side of your index finger.',
  B: 'Flat hand, fingers together and pointing up, thumb folded across the palm.',
  C: 'Curve your hand into the shape of the letter C.',
  D: 'Index finger points up; other fingers and thumb touch to make a small circle.',
  E: 'Fingers curled down onto the thumb, like a tight claw.',
  F: 'Touch index finger and thumb together; the other three fingers stand up.',
  G: 'Index finger and thumb point sideways, parallel, like a small pinch.',
  H: 'Index and middle fingers point sideways together.',
  I: 'Fist with the little finger pointing up.',
  J: 'Make an I, then draw a J in the air with your little finger.',
  K: 'Index and middle fingers up in a V, thumb resting between them.',
  L: 'Index finger up and thumb out, forming an L.',
  M: 'Fist with the thumb tucked under three fingers.',
  N: 'Fist with the thumb tucked under two fingers.',
  O: 'All fingertips meet the thumb to make an O.',
  P: 'Like K, but pointing the hand downward.',
  Q: 'Like G, but pointing the hand downward.',
  R: 'Cross your index and middle fingers.',
  S: 'Fist with the thumb wrapped across the front of the fingers.',
  T: 'Fist with the thumb poking up between index and middle finger.',
  U: 'Index and middle fingers up, held together.',
  V: 'Index and middle fingers up and spread apart (peace sign).',
  W: 'Index, middle and ring fingers up and spread.',
  X: 'Index finger up and hooked like a crooked hook.',
  Y: 'Thumb and little finger out, other fingers curled (shaka).',
  Z: 'Point your index finger and trace a Z in the air.',
};

// Hand counts come from the ISL training photos (share of frames where two hands
// were detected). Tips are only given where the dataset's reference chart is unambiguous.
const islHands: Record<string, 1 | 2 | 'either'> = {
  A: 2, B: 2, C: 1, D: 2, E: 2, F: 2, G: 2, H: 2, I: 1, J: 1, K: 2, L: 1, M: 2,
  N: 'either', O: 1, P: 2, Q: 1, R: 'either', S: 'either', T: 2, U: 1, V: 1, W: 2, X: 2, Y: 2, Z: 2,
};
const islTips: Record<string, string> = {
  A: 'Both hands, index fingers meeting at the top like a small roof.',
  I: 'One hand, index finger pointing straight up.',
  L: 'One hand, index finger up and thumb out to form an L.',
  V: 'One hand, index and middle fingers raised in a V.',
  W: 'Both hands with the fingers interlaced.',
  X: 'Both hands, index fingers crossed to make an X.',
};

const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

export const ASL: LetterInfo[] = letters.map((letter) => ({
  letter,
  hands: 1,
  tip: asl[letter],
  motion: letter === 'J' || letter === 'Z',
}));

export const ISL: LetterInfo[] = letters.map((letter) => ({
  letter,
  hands: islHands[letter],
  tip: islTips[letter] ?? 'Follow the ISL reference chart, then check it in the translator.',
}));
