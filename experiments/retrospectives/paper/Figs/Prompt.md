# Role
You are a world-class academic illustration expert specializing in creating high-quality, intuitive, and aesthetically pleasing paper architecture figures for top-tier computer vision and AI conferences (e.g., CVPR, NeurIPS, ICLR).

# Task
Read the provided [paper method description], fully understand its core mechanism, module composition, and data flow. Then, based on your understanding, design and draw a professional academic architecture figure.

# Visual Constraints
1. Style tone:
   - Must have a top-conference paper style: professional, clean, modern, minimalist.
   - Core aesthetics: use a flat vector illustration style with simple lines, referencing the figure aesthetics in DeepMind or OpenAI papers.
   - Reject cartoonish, painterly, or overly artistic looks; maintain a rigorous academic figure aesthetic.
   - Background must be pure white, with no textures or shadows.
   - 4K resolution, Make the picture look very clear.
   - Use the Times New Roman font.

2. Color system:
   - Strictly use light or soft tones.
   - Avoid overly saturated colors (e.g., bright red or green) or overly dark/heavy colors. Use variations in lightness to distinguish different module types.

3. Content and layout:
   - Convert your understanding of the methodology into clear modules and data-flow arrows.
   - Appropriately use modern, simple vector icons embedded within modules to enhance clarity.

4. Text rules:
   - All text in the figure must be in English.
   - You must add clear, legible text labels for key modules or equations mentioned in the methodology.
   - Do not include long sentences, descriptive paragraphs, or complex formulas in the figure. Text is for identifying modules, not explaining principles.

5. Prohibitions:
   - No photorealistic imagery.
   - No messy sketch lines.
   - No hard-to-read text.
   - No cheap 3D shadow artifacts.

# paper method description
Despite significant progress, applying representation models directly to noisy clinical domains (e.g., dermatology) exposes a fundamental gap. Real-world clinical text is highly unstructured and heterogeneous. The core bottleneck is severe intra-domain semantic heterogeneity. Neither local vocabulary matching nor general contextual understanding can bridge these surface variations to achieve precise "clinical semantic alignment." Specifically, representation learning faces a triple challenge: First, *the limitations of traditional medical text representation models*. Constrained by shorter context windows and limited parameter spaces, traditional models not only frequently lose critical semantics due to forced truncation when processing texts, but also struggle to adequately represent complex medical concepts. Meanwhile, for different downstream tasks, traditional medical encoders typically provide identical representations, lacking task-aware capabilities. Second, *clinical equivalence under surface variation*. Equivalent cases are expressed drastically differently (e.g., a patient's "itchy elbows with flaking" vs. a clinician's "erythematous plaques with scale" share no lexical overlap but identical meanings), requiring semantic invariance across stylistic noise. Third, *semantic divergence under subtle textual differences*. Simply replacing "scale" with "vesicles" leaves the surface form nearly unchanged but completely flips the diagnosis. Due to mismatched training objectives, existing models fail to simultaneously filter stylistic noise and capture subtle diagnostic differences, falling short of sentence-level clinical alignment requirements.