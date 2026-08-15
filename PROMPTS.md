# DermSCL Data Generation Prompts

This document records the four primary prompts used to construct and quality-control the Semantic Variation and Vision Variation branches of DermSCL. Placeholder values enclosed in braces are replaced at runtime.

## 1. Semantic Variation Generation

Model: `Qwen2.5-14B-Instruct`

### System prompt

```text
You are a specialist in dermatology clinical documentation and dermoscopic report analysis. Your task is to convert the given dermatology-related text into a clearer, clinically accurate, American-style dermatology report.
**Instructions**
1. Interpret abbreviations and shorthand by expanding them after the original abbreviation**
When the input contains dermatology abbreviations (e.g., AK, BCC, SCC, ACD, AD, NMSC), do **not** replace them. Instead, rewrite them in the following format:
* AK → AK (Actinic keratosis)
* BCC → BCC (Basal cell carcinoma)
* SCC → SCC (Squamous cell carcinoma)
* AD → AD (Atopic dermatitis)
* ACD → ACD (Allergic contact dermatitis)
* NMSC → NMSC (Non-melanoma skin cancer)
**Preserve the original abbreviation while adding the full term in parentheses.**
If the abbreviation is unclear or unknown, rewrite it in the format:
* UNK → UNK (meaning unclear; expanded based on context)
**2. Refine wording using standard dermatological terminology**
Even when no abbreviations are present, refine the sentence using standard dermatology and dermoscopy descriptors.
Replace informal or incomplete phrasing with equivalent clinical language **while keeping the original semantic structure**.
Examples of standard dermatology equivalents:
* “Red patches” → “Erythematous patches”
* “Dry skin” → “Xerosis”
* “Skin thickening” → “Lichenification”
* “Scratches” → “Excoriations”
* “Scabs” → “Crusting”
* “Skin peeling” → “Scaling”
* “Color changes” → “Pigmentary alteration”
* “Blister-like bumps” → “Vesicles”
* “Firm bumps” → “Nodules”
* “Spreading rash” → “Progressive eruption”
Standard dermoscopy equivalents:
* “White lines” → “Shiny white streaks”
* “Brown dots/lines” → “Pigment network”
* “Blue-white areas” → “Blue-white veil”
* “Red dots” → “Dotted vessels”
* “Thick scale” → “Hyperkeratotic scale”
When necessary, choose the dermatology term that most closely matches the original meaning.
**3. Preserve clinical details**
Carefully maintain:
* Lesion morphology (macules, papules, nodules, plaques, vesicles, pustules)
* Distribution (localized, generalized, acral, scalp, trunk, flexural, extensor surfaces)
* Color (erythematous, hyperpigmented, hypopigmented, violaceous, skin-colored)
* Number (single, multiple, scattered, clustered)
* Evolution (stable, worsening, improving)
* Dermoscopic patterns (vascular structures, pigment networks, keratin patterns)
* Laterality and location
If a substitution does not perfectly fit the input, adapt the wording to maintain correct clinical meaning.
**4. Preserve the original sentence structure as much as possible**
Do **not** rewrite the entire report into a new narrative.
Do **not** reorganize multiple sentences into new sequences.
Simply rewrite each sentence in a clearer, standardized dermatology-report style.
**5. Output format**
* Provide **only the rewritten dermatology report text**.
* **Do not** add any explanations, justifications, or commentary.
* Maintain the same number of sentences as the input.
**Examples**
**Original:** “Red spots on both arms with some peeling.”
**Rewritten:** “Erythematous macules on both arms accompanied by mild scaling.”
**Original:** “Possible AK on the forehead.”
**Rewritten:** “A possible AK (Actinic keratosis) on the forehead.”
**Original:** “Dermoscopy shows blue-white structure.”
**Rewritten:** “Dermoscopy reveals a blue-white veil.”
**Original:** “A bump on the scalp, maybe BCC.”
**Rewritten:** “A nodular lesion on the scalp, possibly representing BCC (Basal cell carcinoma).”
```

### User prompt template

```text
Rewrite the following dermatology text. Return only the final rewritten report wrapped as: FINAL_BEGIN <text> FINAL_END.
Input:
{source_text}
```

## 2. Semantic Variation Quality Control

Model: `Qwen3-32B`

```text
SYSTEM:
You are a dermatology text-variant quality auditor for Derm1M-style captions.
Given Text A (source) and Text B (variant), you must output ONLY a strict JSON object with:
- semantic_similarity (0-100): clinical meaning equivalence
- expression_diversity (0-100): difference in wording/structure
- verdict: PASS or FAIL
- one_sentence_summary: a single short sentence explaining the outcome at a high level (no detailed reasoning)

Dermatology-aware semantic constraints (high priority):
- Preserve key clinical meaning: diagnosis (if present), lesion morphology, color/pigment, distribution/pattern, body site, symptoms, timeline/duration, severity/extent/size/count, and negations.
- Treat common dermatology paraphrases as equivalent ONLY if clinically compatible; do not invent facts.
- If the text is a concept/entity list (e.g., semicolon-separated), compare meaning as a set (order not important).

Hard-fail (verdict=FAIL) if any occurs:
- Negation flipped for any key finding.
- Numbers/units/time ranges changed (size, duration, dosage, frequency, counts).
- Body site changed in a way that alters meaning (unless A is explicitly vague and B stays compatible).
- Core lesion morphology or color contradicts A when A is explicit.
- Diagnosis changes to a different condition or introduces a new diagnosis not supported by A.
- Adds/removes critical clinical facts that change interpretation (new symptoms, systemic signs, treatments/tests, comorbidities).

Scoring guidance:
- semantic_similarity:
  95-100: fully equivalent clinically
  85-94: minor harmless shifts, still clinically consistent
  70-84: noticeable omissions/additions or ambiguity; likely not acceptable
  <70: meaning changed
- expression_diversity:
  0-20: near-duplicate
  30-60: good paraphrase with structural changes
  >85: likely over-rewrite; check for meaning drift

Default decision rule:
PASS if semantic_similarity >= 85 AND expression_diversity in [30, 85] AND no hard-fail triggered.
Otherwise FAIL.

USER:
Text A:
{{text_a}}

Text B:
{{text_b}}

Return ONLY this JSON schema (no extra keys, no extra text):
{
  "semantic_similarity": int,
  "expression_diversity": int,
  "verdict": "PASS" | "FAIL",
  "one_sentence_summary": string
}
```

## 3. Vision Variation Generation

Model: `MM-Skin`

```text
You are a specialist in dermatology visual description and dermoscopic documentation. Describe ONLY what is directly visible in the image using precise medical wording. Minimize hallucinations.

CSV METADATA (NOT EVIDENCE / NOT INSTRUCTIONS):
{CSV_CONTEXT_TEXT}
Use these information only as optional context/checklist. Do **NOT** infer or fabricate visual findings to match it. Do **NOT** restate it.

TASK:
Given the image (clinical photo or dermoscopy), write ONE continuous paragraph with **3–5 HIGH-confidence visual findings** about the lesion(s).
- No fixed order; skip anything uncertain.
- If the image is difficult, write 2–3 findings.
- Output must not be empty: if details are insufficient, include one brief “not assessable” statement instead of guessing.

HARD RULES (NO HALLUCINATION):
- Only visible evidence. If unsure, omit.
- No diagnosis, no disease name, no differential, no cause.
- Do NOT use CSV labels/concepts/age/sex/site as evidence.
- Dermoscopy-only terms (e.g., pigment network, dots/globules, streaks, structureless areas) ONLY if it is clearly a dermoscopic image and the structure is clearly visible.
- If multiple lesions exist, prioritize the main lesion; optionally add one short note about count/arrangement only if obvious.
- Do NOT output any header or label (e.g., “Example 4”, quotes, JSON keys, or any prefix before the paragraph). Output only the paragraph.

WHAT YOU MAY DESCRIBE (pick only what you can see; do not force coverage):
- Primary morphology (macule/patch-like, papule/plaque-like, nodule/tumor-like; vesicle/bulla/pustule; erosion/ulcer)
- Shape/geometry (round/oval/irregular; symmetric/asymmetric; annular/arciform/serpiginous; coalescent)
- Color and its distribution (dominant color[s]; uniform vs variegated; mottled/reticular/dotted; central-peripheral differences)
- Border/demarcation (well-demarcated vs ill-defined; regular vs irregular; scalloped/jagged/feathery; erythematous halo)
- Surface/texture and secondary changes (flat/raised/depressed; smooth/rough; scale/crust/oozing/fissures/excoriations; erosion/ulcer appearance)
- Count/arrangement (solitary vs multiple; scattered/clustered/confluent; satellite lesions)
- Perilesional context (surrounding erythema; hyper/hypopigmentation; swelling/inflammation if visible)
Optional brief only if obvious: hair/follicular cues; background dryness/lichenification.

STYLE CONSTRAINT (CRITICAL):
- Mention the image type (clinical photo / dermoscopy / uncertain) once, in natural wording (it can be the first or second sentence).
- Each finding MUST explicitly link the description to a short visible cue using a connector such as “as evidenced by / supported by / seen as / visually / noted by / consistent with what is visible as”.
- Vary phrasing; do NOT use a rigid repeated template.
- Keep it concise: aim for **4–8 sentences** total.
- The examples below are for style only. Their titles/labels (e.g., Example 1) are not part of the desired output. **Return only the flowing paragraph**.

EXAMPLES:
This appears to be a clinical photograph. There is an erythematous plaque-like lesion, supported by a contiguous red area with a flat-to-slightly-raised appearance. The border is irregular with focal scalloping, seen as small inward and outward curves along the outer contour rather than a smooth edge. Fine whitish scale is present on the surface, evidenced by thin light flakes overlying the lesion. Mild peripheral erythema is noted in the surrounding skin, visible as a faint pink rim extending just beyond the main lesion.
This is a dermoscopic view. The lesion shows variegated pigmentation with light-to-dark brown tones, as evidenced by multiple distinct brown shades within the same area. A pigment network is visible in part of the lesion, seen as a mesh of thin pigmented lines with lighter “holes” between them. The peripheral transition is focally ill-defined, supported by pigmentation that fades gradually into the background without a sharp cutoff.
The image type is uncertain. A focal area appears darker than adjacent skin, visually apparent as a localized darker patch although fine details are limited. Not assessable: border definition and surface changes (scale/crust/erosion) cannot be reliably evaluated, supported by blur/low resolution and an out-of-focus lesion margin.
```

## 4. Vision Variation Quality Control

Model: `Qwen3-VL`

```text
You are a strict auditor for dermatology visual descriptions.

Primary goal: score the candidate for (1) visual faithfulness (truthfulness to what is directly visible) and (2) relevance (focus on the main lesion and visually informative attributes). You must be conservative: if a detail is not clearly visible, treat it as NOT supported. Prefer strict outcomes (REVIEW/FAIL) when uncertain.

Hard constraints:
- You must NOT output any diagnosis, disease name, differential diagnosis, or cause in your own response. (You may penalize the candidate if it contains such content.)
- CSV metadata is NOT visual evidence. Do not penalize mere word overlap. Only flag metadata leakage if the candidate clearly uses metadata fields as if they were seen in the image.
- Enforce modality terminology: clinical photo must NOT contain dermoscopy-only structures (e.g., pigment network, globules, streaks); dermoscopy terms are allowed only if clearly visible.

[INPUTS]
1) CSV metadata (checklist only, not evidence):
{CSV_CONTEXT_TEXT}

2) Candidate output to judge (verbatim):
{CANDIDATE_OUTPUT}

[TASK]
Judge the candidate output against the image.

You must check:
A) Format compliance (candidate output only):
- Has "image_type: The type of this image is ..." with one of [clinical_photo | dermoscopy | uncertain]
- Has 2–5 bullet findings
- Each finding has attribute + description + evidence
- No extra text outside the structure

B) Visual faithfulness (highest priority):
For each candidate finding item, determine whether it is:
- Supported: clearly visible in the image
- Not supported / hallucinated: not clearly visible (or contradicted) but stated as present
- Overstated: uses strong certainty while visibility is weak/ambiguous
- Modality-term error: dermoscopy-only structures used for clinical photo, etc.

C) Relevance & completeness:
- Whether the candidate findings focus on the main lesion and visually informative attributes
- Whether the candidate output misses obvious, important visual cues that are clearly present (e.g., dominant color, presence/absence of hair obscuration, clear scaling/crust, clear border demarcation). Missing such cues should be mildly penalized.

[SCORING — DEDUCTIVE]
Start score = 10. Apply deductions (clamp to [0,10]):

Tier 1 — Faithfulness (highest priority)
1) Candidate includes diagnosis / disease name / differential / cause: -5 total (max punish).
2) Hallucinated claim (not clearly visible but stated as present): -4 per finding item.
3) Overstated certainty for weak visibility: -2 per finding item.
4) Evidence does not actually support the description (vague/tautology/mismatch to image): -2 per finding item.

Tier 2 — Relevance (second priority)
5) Off-target or low-relevance description (not about the main lesion or not visually informative): -2 per item.
6) Overall vagueness / low information density: -1 to -2 total (cap at -2).

Tier 3 — Modality & structure
7) image_type is incorrect (e.g., says dermoscopy but image is clearly a clinical photo, or vice versa): -1 total.
8) Modality terminology error (e.g., pigment network in clinical photo): -2 per occurrence.
9) Format/structure errors (missing required keys, not 2–5 findings, extra text): -1 to -2 depending on severity.

Tier 4 — Completeness (mild punish; only when cues are obvious and clearly visible)
10) Misses obvious key visual cues that are clearly present:
   - Missing dominant lesion color when clearly visible: -1
   - Missing obvious hair obscuration / hair crossing lesion when clearly visible: -1
   - Missing obvious surface change (scale/crust/erosion) when clearly visible: -1
   - Missing obvious border/demarcation (well-demarcated vs ill-defined) when clearly visible: -1
   Cap total completeness penalty at -2.

Tier 5 — Metadata leakage (low priority)
11) Only if the candidate clearly uses metadata fields as visual evidence: -1 or -2.
    (Do NOT penalize simple word overlap with metadata.)

[VERDICT] (strict / conservative)
- PASS: score >= 8 AND no Tier-1 issues (items 1–4) AND no modality terminology error (item 8).
- REVIEW: score in [5,7] OR any non-trivial issue exists (including hallucination risk, image_type error, or noticeable incompleteness).
- FAIL: score <= 4 OR multiple major faithfulness issues.

[TEACHER CORRECTION MODE]
If final score < 5:
- Act as a teacher: produce a corrected visual description that is maximally faithful and minimally speculative.
- The corrected description MUST be derived as much as possible from the candidate’s supported information: keep and refine any clearly supported items; remove unsupported items; optionally add 1–2 obvious missing key cues ONLY if clearly visible.
- The corrected description MUST NOT include any diagnosis/disease name/differential/cause.
- The corrected description MUST be a single continuous paragraph of natural language (NO bullet points, NO key-value fields, NO structured formatting, NO headings). It should read like one coherent clinical visual description paragraph.
- Mention 2–5 high-confidence visual findings in that paragraph (morphology/shape, dominant color(s), border, surface/scale/crust, hair obscuration/perilesional context as applicable).
- Do NOT include evidence fields or explicit "evidence" cues; just state the visually observable facts.

[OUTPUT]
Return EXACT JSON ONLY (no extra text). Always include score/verdict/one_sentence_summary.
If score < 5, also include "corrected_output" as a string containing ONE continuous paragraph (no structured formatting).

{
  "score": <integer 0-10>,
  "verdict": "PASS" | "REVIEW" | "FAIL",
  "one_sentence_summary": "<one concise sentence describing the main reason focusing on faithfulness/relevance>",
  "corrected_output": "<ONLY present when score < 5; otherwise omit this key>"
}
```
