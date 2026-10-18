// The fixed catalog of ~101 sentences the recognizer knows, per sign language. Both files
// below are fetched at runtime from the language's model folder (`base`: '/sentence_model'
// for ISL, '/sentence_model_asl' for ASL), not hardcoded here:
//   <base>/labels.json  -- class index -> sentence text, written by the same export script
//                          as the DTW bank so it can never drift out of sync with it.
//   <base>/glosses.json -- sentence text -> sign-gloss word order (ISL: from the ISL-CSLTR
//                          dataset's own annotation file; ASL: model_assets/asl_sentences/catalog.json).

export interface CatalogEntry {
  index: number;
  sentence: string;
  gloss: string | null;
}

export async function loadCatalog(base = '/sentence_model'): Promise<CatalogEntry[]> {
  const [labels, glosses] = await Promise.all([
    fetch(`${base}/labels.json`).then((r) => {
      if (!r.ok) throw new Error(`Failed to fetch sentence labels (${r.status})`);
      return r.json() as Promise<string[]>;
    }),
    fetch(`${base}/glosses.json`)
      .then((r) => (r.ok ? (r.json() as Promise<Record<string, string>>) : ({} as Record<string, string>)))
      .catch(() => ({}) as Record<string, string>),
  ]);
  return labels.map((sentence, index) => ({ index, sentence, gloss: glosses[sentence] ?? null }));
}
