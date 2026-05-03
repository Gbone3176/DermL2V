DERM_EMBEDDING_PROMPTS = {
    "SemVariants": [
        "Read the provided dermatological condition description and return the candidate description that matches its meaning most closely.",
        "Given a skin-condition description, select the candidate description with the highest meaning-level similarity.",
        "Match the input dermatology description to the closest candidate description by semantics rather than exact wording."
    ],
    "VisVariants": [
        "Given a diagnosis-style dermatology text, retrieve the visual-description text that best matches it in meaning.",
        "Using the provided dermatology diagnostic statement as input, select the visual-description passage that is most relevant."
    ],
    "DermQA": [
        "Given a dermatology-related question, select the answer that is most relevant to what the question is asking.",
        "For a question in the skin-disease domain, return the candidate answer with the highest semantic relevance."
    ],
    "SI1": [
        "Retrieve the most appropriate answer for this dermatology question.",
        "Retrieve the answer entry that best matches the clinical vignette and question."
    ],
    "SI2": [
        "Given a dermatology question, retrieve the single most relevant and most correct answer passage that directly answers it.",
        "Find the dermatology answer passage that best matches this question and provides the highest-correctness response."
    ],
}
