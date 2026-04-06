# -*- coding: utf-8 -*-
"""
experimental_documentation.py

Supplementary Documentation for:
"Medical or General-Purpose LLMs for Virtual Patient Simulation?
 A Comparative Evaluation in OSCE-Aligned Scenarios"

Published at: IEEE SeGAH 2026
Author: Abdulrahman Aloraini, Qassim University

This file provides complete transparency on experimental design,
prompting strategy, evaluation metric definitions, and scoring
algorithms used in the study. It addresses reviewer feedback
requesting reproducibility details.

Contents:
    1. System Prompt Template (verbatim)
    2. Generation Parameters
    3. Symptom Injection & Persona Generation
    4. Examiner Question Design
    5. Difficulty Level Operationalization
    6. Factorial Design & Persona Sampling
    7. Practical Metric Scoring Algorithms (formal definitions)
    8. Literature-Based Metric Scoring Algorithms
    9. Metric Weighting Justification (OSCE mapping)
   10. Human Evaluation Sampling Strategy

================================================================================
"""

# ============================================================================
# 1. SYSTEM PROMPT TEMPLATE (VERBATIM)
# ============================================================================
#
# This is the exact system prompt injected into every LLM conversation.
# All placeholders are filled programmatically from the patient persona
# (see Section 3 for how personas are generated).
#
# Key design decisions:
#   - The model is instructed to BE the patient, not simulate one
#   - All symptoms are explicitly provided (no reliance on model's medical knowledge)
#   - Negative symptoms are listed so the model knows what to DENY
#   - Health literacy level controls language complexity
#   - Personality and emotional state guide conversational style

SYSTEM_PROMPT_TEMPLATE = """You are role-playing as a patient in a medical training simulation.

## YOUR IDENTITY
You ARE {name}, a {age}-year-old {gender} {occupation}. This is REAL to you.
{special_note}

## CRITICAL RULES
1. NEVER say you're an AI/simulation/language model
2. NEVER use medical jargon (your education: {literacy})
3. NEVER diagnose yourself
4. NEVER report symptoms you don't have
5. DENY symptoms in your "negative" list when asked

## YOUR SITUATION
Chief Complaint: "{complaint}"
Started: {onset} | Trigger: {trigger}

## YOUR SYMPTOMS
{symptoms}

## SYMPTOMS YOU DON'T HAVE (deny if asked)
{negative}

## BACKGROUND
- History: {history}
- Medications: {meds}
- Allergies: {allergies}
- Smoking: {smoking}

## PERSONALITY
- Emotional: {emotional}
- Communication: {personality}
- Pain level: {pain}/10
- Style: {verbosity}

Respond naturally as {name}. Stay in character."""

# For pediatric scenarios, {special_note} is set to:
#   "NOTE: You are the PARENT ({caregiver_name}) speaking for your child {name}."
# For geriatric scenarios:
#   "NOTE: You may have some baseline cognitive slowing."
# Otherwise it is empty.


# ============================================================================
# 2. GENERATION PARAMETERS
# ============================================================================
#
# All models used identical generation parameters to ensure fair comparison.
# These were held constant across all 576 conversations.

GENERATION_CONFIG = {
    "temperature": 0.7,       # Balances creativity with consistency
    "max_tokens": 250,        # Sufficient for natural patient responses
    "top_p": 1.0,             # Default (no nucleus sampling restriction)
    "repetition_penalty": 1.0, # Default (handled by temperature)
    "turns_per_conversation": 10,  # Fixed 10-turn dialogues
    "quantization": "4-bit (bitsandbytes)",  # For all models except OpenBioLLM (bfloat16)
}

# Model-specific prompt formatting was applied per architecture:
#   - LLaMA-3/3.1 models: Official Llama-3 chat template with <|begin_of_text|>,
#     <|start_header_id|>, <|eot_id|> tokens
#   - Mistral models: [INST] / [/INST] template (system merged into first user turn)
#   - Qwen2.5: ChatML format with <|im_start|> / <|im_end|> tokens
#
# See MODEL_PROMPT_TEMPLATES below for exact formatting per model.

MODEL_PROMPT_TEMPLATES = {
    "llama3": {
        "applies_to": [
            "MedQA-Llama3-8B",
            "Bio-Medical-Llama3-8B",
            "Medical-Llama3-8B",
            "OpenBioLLM-8B",
            "LLaMA-3.1-8B-Instruct",
        ],
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant": "{message}<|eot_id|>",
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "mistral": {
        "applies_to": [
            "Mistral-7B-Instruct-v0.3",
            "Mistral-Nemo-12B-Instruct",
        ],
        "system": "",  # System prompt merged into first [INST] block
        "user": "[INST] {message} [/INST]",
        "assistant": "{message}</s>",
        "stop_tokens": ["</s>"],
    },
    "chatml": {
        "applies_to": ["Qwen2.5-7B-Instruct"],
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n",
        "assistant": "{message}<|im_end|>\n",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
    },
}


# ============================================================================
# 3. SYMPTOM INJECTION & PERSONA GENERATION
# ============================================================================
#
# IMPORTANT (addresses Reviewer 3's concern):
#   Full symptom profiles are EXPLICITLY INJECTED into the system prompt.
#   Models do NOT rely on their own medical training to generate symptoms.
#   Every symptom, its severity, natural-language variation, and negative
#   symptoms to deny are provided directly in the prompt.
#
# Persona generation is RANDOMIZED per factorial cell. For each combination
# of (model × scenario × difficulty × persona_slot), a unique patient is
# generated with randomized:
#   - Name, age (within scenario-appropriate range), gender
#   - Occupation
#   - Which non-primary symptoms are present (probabilistic based on prevalence)
#   - Specific symptom variation wording (randomly selected)
#   - Medical history and medications (scenario-appropriate, probabilistic)
#   - Allergies, smoking status
#   - Emotional state and personality (difficulty-dependent)
#   - Health literacy and verbosity (difficulty-dependent)
#
# Example of how a symptom is injected into the prompt:
#   "- pressure in my chest (severity: 8/10)"
#   "- sweating badly (severity: 6/10)"
#   "- can't catch breath (severity: 5/10)"
#
# The system prompt then lists what to DENY:
#   "fever, cough, sharp pleuritic pain, rash"

SYMPTOM_INJECTION_EXAMPLE = """
=== Example: Chest Pain ACS, Moderate Difficulty ===

System prompt excerpt (symptoms section):
  ## YOUR SYMPTOMS
  - pressure in my chest (severity: 8/10)
  - cold sweats (severity: 6/10)
  - hard to breathe (severity: 5/10)
  - feel sick (severity: 4/10)

  ## SYMPTOMS YOU DON'T HAVE (deny if asked)
  fever, cough, sharp pleuritic pain, rash

The model sees EXACTLY what symptoms to report and what to deny.
It cannot fabricate symptoms beyond what is provided.
"""

# Symptom prevalence is adjusted by difficulty:
#   Easy:     non-primary prevalence × 1.3 (more symptoms present → easier diagnosis)
#   Moderate: default prevalence values
#   Hard:     non-primary prevalence × 0.6 (fewer symptoms present → harder diagnosis)
#             Primary symptoms are always present regardless of difficulty.


# ============================================================================
# 4. EXAMINER QUESTION DESIGN
# ============================================================================
#
# For each of the 8 clinical scenarios, we created 3 sets of 10 scripted
# examiner questions corresponding to 3 difficulty levels:
#
#   Easy (10 questions):     Structured, specific, closed-ended clinical questions
#                            that guide the student step-by-step through history-taking.
#                            Example: "Can you describe the chest pain for me?
#                            What does it feel like?"
#
#   Moderate (10 questions): Semi-structured, moderately specific questions
#                            that require some clinical reasoning to follow up.
#                            Example: "Tell me more about what you're feeling."
#
#   Hard (10 questions):     Open-ended, vague questions that demand the student
#                            (and the VP) manage ambiguity and volunteer information
#                            appropriately.
#                            Example: "So what's going on today?"
#
# Design rationale:
#   - Questions were authored to mirror real OSCE examiner styles at each level
#   - Easy questions follow textbook history-taking structure (HPI → ROS → PMH → FH)
#   - Hard questions test whether the VP can provide coherent, unprompted disclosure
#   - All questions are scenario-specific (e.g., pediatric questions address the parent)
#   - The same question set is used for all models within a (scenario, difficulty) cell
#
# Total unique question sets: 8 scenarios × 3 difficulties = 24 sets of 10 questions

EXAMINER_QUESTION_EXAMPLES = {
    "chest_pain_acs": {
        "easy_sample": [
            "Hello, what brings you to the emergency room today?",
            "Can you describe the chest pain for me? What does it feel like?",
            "Where exactly do you feel the pain?",
            "When did the pain start, and how long has it been going on?",
            "On a scale of 1 to 10, how bad is the pain right now?",
        ],
        "moderate_sample": [
            "What brings you in today?",
            "Tell me more about what you're feeling.",
            "Can you point to where it hurts?",
            "When did this all start?",
            "How would you rate the pain?",
        ],
        "hard_sample": [
            "So what's going on today?",
            "Can you tell me more about that?",
            "Anything else?",
            "How have you been feeling otherwise?",
            "What have you tried so far?",
        ],
    },
}


# ============================================================================
# 5. DIFFICULTY LEVEL OPERATIONALIZATION
# ============================================================================
#
# Difficulty affects BOTH the patient persona AND the examiner questions.
#
# Patient-side adjustments:
#
#   +-----------------+------------------+-------------------+-------------------+
#   | Parameter       | Easy             | Moderate          | Hard              |
#   +-----------------+------------------+-------------------+-------------------+
#   | Emotional state | calm             | anxious/worried   | very_anxious /    |
#   |                 |                  |                   | irritable / stoic |
#   | Personality     | cooperative      | cooperative       | poor_historian /  |
#   |                 |                  |                   | tangential /      |
#   |                 |                  |                   | minimizer         |
#   | Health literacy | high             | moderate          | low               |
#   | Verbosity       | normal           | normal            | brief or verbose  |
#   | Symptom prev.   | ×1.3 (more       | default           | ×0.6 non-primary  |
#   |                 | symptoms shown)  |                   | (fewer clues)     |
#   +-----------------+------------------+-------------------+-------------------+
#
# Examiner-side adjustments:
#   Easy:     Specific, structured questions that scaffold history-taking
#   Moderate: Semi-structured questions requiring some inference
#   Hard:     Open-ended, vague questions demanding VP-driven disclosure


# ============================================================================
# 6. FACTORIAL DESIGN & PERSONA SAMPLING
# ============================================================================
#
# Full factorial design: 8 models × 8 scenarios × 3 difficulties × 3 personas
#                      = 576 unique conversations
#
# PERSONA SAMPLING STRATEGY:
#   For each (scenario, difficulty) cell, 3 unique personas are generated
#   using different random seeds. The SAME 3 personas are used across all
#   8 models for that cell, ensuring that model comparisons within each
#   (scenario, difficulty, persona_slot) triple are controlled.
#
#   This means:
#     - 8 scenarios × 3 difficulties × 3 personas = 72 unique patient personas
#     - Each persona is evaluated by all 8 models
#     - Randomization covers: name, age, gender, occupation, symptom subset,
#       symptom variation wording, medical history, medications, allergies,
#       smoking status, emotional state, personality

FACTORIAL_DESIGN = {
    "models": 8,
    "scenarios": 8,
    "difficulty_levels": 3,       # easy, moderate, hard
    "personas_per_cell": 3,       # 3 randomly generated personas per (scenario, difficulty)
    "turns_per_conversation": 10,
    "total_conversations": 576,   # 8 × 8 × 3 × 3
    "total_unique_personas": 72,  # 8 × 3 × 3 (shared across models)
}

CLINICAL_SCENARIOS = {
    "chest_pain_acs":      "Chest Pain – Acute Coronary Syndrome (Cardiology, emergent)",
    "pneumonia":           "Community-Acquired Pneumonia (Pulmonology, urgent)",
    "diabetes":            "Type 2 Diabetes – Poor Control (Endocrinology, routine)",
    "depression":          "Major Depressive Episode (Psychiatry, urgent)",
    "appendicitis":        "Acute Appendicitis (Surgery, emergent)",
    "pediatric_fever":     "Pediatric Febrile Illness (Pediatrics, urgent, proxy historian)",
    "geriatric_confusion": "Acute Confusion in Elderly / Delirium (Geriatrics, urgent)",
    "anaphylaxis":         "Anaphylactic Reaction (Emergency, critical, time-critical)",
}


# ============================================================================
# 7. PRACTICAL METRIC SCORING ALGORITHMS (FORMAL DEFINITIONS)
# ============================================================================
#
# Each metric scores a 10-turn conversation transcript on [0, 1].
# Below we provide the exact scoring logic, equivalent to pseudocode.
# Full implementation is in virtualpatient_fixed_rerunonly.py.
#
# Notation:
#   R = list of 10 patient (assistant) responses
#   Q = list of 10 examiner (user) questions
#   P = patient persona object
#   T = concatenation of all responses (lowercased)

PRACTICAL_METRICS = {
    "role_fidelity": {
        "weight": 0.20,
        "description": "Does the VP maintain patient character without breaking immersion?",
        "reference": "Talbot et al. (2012)",
        "algorithm": """
            score = 1.0
            for pattern in AI_BREAKING_PATTERNS:   # 10 patterns: "as an ai", "language model", etc.
                if pattern found in T:
                    score -= 0.4                    # Severe penalty per AI-break
            if patient's first name appears in T:
                score += 0.05                       # Small bonus for name use
            return clamp(score, 0, 1)
        """,
        "ai_breaking_patterns": [
            "as an ai", "i am an ai", "language model", "i cannot provide medical",
            "i'm programmed", "in this simulation", "i don't have feelings",
            "as a virtual", "i was designed", "my training data",
        ],
        "score_range": "[0, 1]. A single AI-break drops score to ≤0.6.",
    },

    "clinical_authenticity": {
        "weight": 0.20,
        "description": "Are symptoms accurate, properly denied, and expressed in lay language?",
        "reference": "Huwendiek et al. (2009)",
        "algorithm": """
            score = 1.0
            for neg in P.negative_symptoms:
                if neg mentioned in T without denial context:
                    score -= 0.08                   # Penalty for undenied negative symptom
            for sym in P.symptoms:
                if any variation of sym found in T:
                    score += 0.02                   # Bonus for natural symptom expression
            return clamp(score, 0, 1)
        """,
        "denial_patterns": "Checks for 'no {word}', 'don't have {word}', 'not {word}', etc.",
        "score_range": "[0, 1]. Typical range: 0.85–1.0 for well-behaved models.",
    },

    "communication_quality": {
        "weight": 0.18,
        "description": "Does the VP respond relevantly to questions with appropriate length?",
        "reference": "Calgary-Cambridge Guide (Silverman et al., 2013)",
        "algorithm": """
            for each (question Q_i, response R_i):
                turn_score = 0.5                    # Baseline
                if Q_i contains a question type keyword (when/where/how/what/scale)
                   AND R_i contains expected response patterns:
                    turn_score += 0.3               # Question-response alignment
                word_count = len(R_i.split())
                if 15 <= word_count <= 80:
                    turn_score += 0.2               # Appropriate length
                elif word_count < 5:
                    turn_score -= 0.2               # Too terse
                elif word_count > 150:
                    turn_score -= 0.1               # Too verbose
                turn_score = min(turn_score, 1.0)
            return mean(all turn_scores)
        """,
        "question_response_patterns": {
            "when": ["ago", "yesterday", "today", "morning", "night", "hour", "day", "week"],
            "where": ["here", "chest", "arm", "back", "stomach", "head", "side"],
            "how": ["like", "feels", "kind of", "sort of", "about", "maybe"],
            "what": ["it's", "i have", "there's", "been", "taking"],
            "scale": [r"\\d+", "out of", "maybe", "about", "pretty"],
        },
        "score_range": "[0, 1]. Hardest metric overall; typical range: 0.55–0.80.",
    },

    "educational_value": {
        "weight": 0.15,
        "description": "Does the VP provide appropriate diagnostic challenge?",
        "reference": "Kononowicz et al. (2019)",
        "algorithm": """
            score = 0.7                             # Baseline
            for phrase in SELF_DIAGNOSIS_PHRASES:    # "i think i have", "i googled", etc.
                if phrase in T:
                    score -= 0.15                   # Ruins educational value
            cv = std(response_lengths) / mean(response_lengths)
            if 0.2 <= cv <= 0.8:
                score += 0.1                        # Healthy variation = progressive disclosure
            if P.difficulty == "hard":
                if hedging markers in T:            # "i'm not sure", "maybe"
                    score += 0.1                    # Appropriate for hard cases
            return clamp(score, 0, 1)
        """,
        "self_diagnosis_phrases": [
            "i think i have", "i probably have", "it might be", "could be",
            "i googled", "i read online", "i diagnosed myself",
        ],
        "score_range": "[0, 1]. Typical range: 0.70–0.85.",
    },

    "linguistic_naturalness": {
        "weight": 0.12,
        "description": "Does the VP speak like a real person, not a clinical textbook?",
        "reference": "Talbot et al. (2012)",
        "algorithm": """
            score = 0.7                             # Baseline
            jargon_count = count of CLINICAL_JARGON terms in T
            if P.health_literacy != "high":
                score -= jargon_count × 0.1         # Patients shouldn't use jargon
            natural_count = count of NATURAL_MARKERS in T
            score += min(0.2, natural_count × 0.03) # Bonus for hedging, fillers
            if bullet points or markdown in T:
                score -= 0.25 (bullets) or 0.15 (markdown)
            return clamp(score, 0, 1)
        """,
        "clinical_jargon_terms": [
            "myocardial infarction", "diaphoresis", "dyspnea", "substernal",
            "etiology", "differential diagnosis", "pathophysiology", "prognosis",
            "bilateral", "unilateral", "proximal", "distal",
            "presenting complaint", "chief complaint", "history of present illness",
        ],
        "natural_markers": [
            "i think", "maybe", "i'm not sure", "kind of", "sort of",
            "you know", "like", "um", "uh", "well", "honestly",
            "i guess", "probably", "it feels like", "i don't know",
        ],
        "score_range": "[0, 1]. Highest validated correlation with human scores (ρ=0.707).",
    },

    "emotional_authenticity": {
        "weight": 0.10,
        "description": "Are emotions contextually appropriate and persona-consistent?",
        "reference": "Calgary-Cambridge Guide",
        "algorithm": """
            score = 0.6                             # Baseline
            emotion_count = count of EMOTIONAL_MARKERS in T
            score += min(0.3, emotion_count × 0.08) # Bonus for emotional expression
            expected = EXPECTED_EMOTIONS[P.emotional_state]
            if any expected emotion found in T:
                score += 0.15                       # Matches persona
            if P.emotional_state == "stoic" and emotion_count <= 2:
                score += 0.1                        # Stoic patients are correctly flat
            return clamp(score, 0, 1)
        """,
        "emotional_markers": [
            "worried", "scared", "nervous", "anxious", "afraid",
            "hope", "concerned", "uncomfortable", "embarrassed",
            "frustrated", "confused", "relieved",
        ],
        "expected_emotions_by_state": {
            "anxious": ["worried", "nervous", "scared", "anxious"],
            "very_anxious": ["terrified", "panicking", "really scared", "very worried"],
            "calm": ["okay", "fine", "alright"],
            "irritable": ["frustrated", "annoyed"],
            "stoic": [],  # Low emotion count is correct
        },
        "score_range": "[0, 1]. Typical range: 0.60–0.90.",
    },

    "dialogue_coherence": {
        "weight": 0.05,
        "description": "Does the VP maintain a consistent narrative without contradictions?",
        "reference": "Conversation Analysis literature",
        "algorithm": """
            score = 0.7                             # Baseline
            for consecutive response pairs (R_i, R_{i+1}):
                prev_words = content words of R_i
                curr_words = content words of R_{i+1}
                overlap = |prev_words ∩ curr_words| / |prev_words|
            avg_overlap = mean(all overlaps)
            if 0.05 <= avg_overlap <= 0.40:
                score += 0.2                        # Healthy topical continuity
            return clamp(score, 0, 1)
        """,
        "note": "Common stop words removed before overlap calculation.",
        "score_range": "[0, 1]. Quality floor metric; typical range: 0.70–0.90.",
    },
}

# Overall practical score formula:
# score = Σ (metric_i × weight_i) for i in 7 metrics
# Weights sum to 1.0 (0.20+0.20+0.18+0.15+0.12+0.10+0.05 = 1.00)

# Additionally tracked but NOT included in the weighted score:
# - Safety/Boundary Compliance: Penalizes treatment recommendations,
#   medication suggestions, self-diagnosis, and medical advice from the VP.
# - Average Response Latency (ms): Generation speed per turn.


# ============================================================================
# 8. LITERATURE-BASED METRIC SCORING ALGORITHMS
# ============================================================================
#
# These 6 metrics use stricter, research-grade criteria grounded in
# validated instruments. They score on [0, 1] but typically yield
# lower values (0.1–0.6) than practical metrics.

LITERATURE_METRICS = {
    "rias_socio_emotional_ratio": {
        "description": "Proportion of socio-emotional vs total coded utterances (RIAS)",
        "reference": "Roter & Larson (2002)",
        "algorithm": """
            For each response, count keyword matches in 5 socio-emotional categories:
              positive_affect:     ["thank", "appreciate", "glad", "happy", ...]
              negative_affect:     ["worried", "scared", "afraid", "anxious", ...]
              empathy_seeking:     ["understand", "know how", "feel like", ...]
              reassurance_seeking: ["will I be", "is it serious", "should I worry", ...]
              social_rapport:      ["how are you", "nice to meet", "thank you doctor", ...]
            And 5 task-focused categories:
              symptom_disclosure, history_giving, medication_info,
              lifestyle_info, clarification
            ratio = socio_emotional_count / total_coded_elements
        """,
        "human_benchmark": "Human physicians typically score 1.0–1.3 SE ratio (Johnson et al., 2004)",
    },

    "portrayal_fidelity": {
        "description": "Composite of symptom coverage + temporal consistency + character consistency",
        "reference": "Howley & Martindale (2004); Barrows (1993)",
        "algorithm": """
            symptom_coverage = (expected symptoms mentioned) / (total expected symptoms)
            temporal_consistency = check for temporal keywords and absence of contradictions
            character_consistency = 1.0 if difficulty-appropriate behavior, else 0.5
            portrayal_fidelity = mean(symptom_coverage, temporal_consistency, character_consistency)
        """,
    },

    "communication_authenticity": {
        "description": "Balance of RIAS affect, dialogue variety, and empathy expression",
        "algorithm": """
            communication_authenticity = mean(
                min(rias_socio_emotional_ratio × 2, 1.0),
                dialogue_variety_score,
                empathy_expression_score
            )
        """,
    },

    "dialogue_variety": {
        "description": "Diversity of dialogue act types (Stolcke et al., 2000)",
        "reference": "Stolcke et al. (2000)",
        "algorithm": """
            Classify each response into dialogue act types:
              statement, yes_no_answer, elaboration, hedge, question,
              emotional_expression, acknowledgment
            variety_score = number_of_unique_act_types / total_possible_types
        """,
    },

    "empathy_expression": {
        "description": "Emotional disclosure and trust-building (CARE-inspired)",
        "reference": "Mercer et al. (2004)",
        "algorithm": """
            Count indicators across 5 categories:
              emotional_disclosure, vulnerability_expression, trust_building,
              help_seeking, reassurance_receptivity
            empathy_score = min(total_indicators / (num_responses × 2), 1.0)
        """,
    },

    "overall_literature_quality": {
        "description": "Weighted mean of all literature-based components",
        "algorithm": """
            overall = mean(
                portrayal_fidelity,
                communication_authenticity,
                min(total_rias_coded / 20, 1.0)     # Content richness
            )
        """,
    },
}


# ============================================================================
# 9. METRIC WEIGHTING JUSTIFICATION (OSCE MAPPING)
# ============================================================================
#
# Weights are calibrated to standard OSCE rubric domain proportions as
# documented in Khan et al. (2013) and Boursicot & Roberts (2005).
#
# Mapping from OSCE domains to our metric weights:
#
#   +---------------------------+----------------+---------------------------+---------+
#   | OSCE Domain               | OSCE Allocation| Our Metrics               | Weight  |
#   +---------------------------+----------------+---------------------------+---------+
#   | Clinical Content          | 35–40%         | Role Fidelity (0.20)      |         |
#   | (history, symptoms,       |                | + Clinical Authenticity    | = 0.40  |
#   |  accuracy)                |                |   (0.20)                  |         |
#   +---------------------------+----------------+---------------------------+---------+
#   | Communication Skills      | 15–20%         | Communication Quality     | = 0.18  |
#   | (question handling,       |                |   (0.18)                  |         |
#   |  structure)               |                |                           |         |
#   +---------------------------+----------------+---------------------------+---------+
#   | Clinical Reasoning /      | 10–15%         | Educational Value         | = 0.15  |
#   | Educational Challenge     |                |   (0.15)                  |         |
#   +---------------------------+----------------+---------------------------+---------+
#   | Rapport / Empathy /       | 10–15%         | Linguistic Naturalness    |         |
#   | Patient Interaction       |                |   (0.12) + Emotional      | = 0.22  |
#   |                           |                |   Authenticity (0.10)     |         |
#   +---------------------------+----------------+---------------------------+---------+
#   | Baseline Quality Floor    | ~5%            | Dialogue Coherence (0.05) | = 0.05  |
#   +---------------------------+----------------+---------------------------+---------+
#   | TOTAL                     | ~100%          |                           | = 1.00  |
#   +---------------------------+----------------+---------------------------+---------+
#
# Robustness check:
#   Under equal weighting (1/7 ≈ 0.143 per metric), the top-4 model cluster
#   remained identical and the rank-order Spearman correlation between weighted
#   and equal-weighted composites was ρ = 0.976 (p < 0.001).

WEIGHT_MAPPING = {
    "osce_clinical_content_35_40pct": {
        "metrics": ["role_fidelity", "clinical_authenticity"],
        "weights": [0.20, 0.20],
        "total": 0.40,
        "references": ["Khan et al. (2013) AMEE Guide 81", "Boursicot & Roberts (2005)"],
    },
    "osce_communication_15_20pct": {
        "metrics": ["communication_quality"],
        "weights": [0.18],
        "total": 0.18,
        "references": ["Calgary-Cambridge Guide (Silverman et al., 2013)"],
    },
    "osce_clinical_reasoning_10_15pct": {
        "metrics": ["educational_value"],
        "weights": [0.15],
        "total": 0.15,
        "references": ["Kononowicz et al. (2019)"],
    },
    "osce_rapport_empathy_10_15pct": {
        "metrics": ["linguistic_naturalness", "emotional_authenticity"],
        "weights": [0.12, 0.10],
        "total": 0.22,
        "references": ["CARE Measure (Mercer et al., 2004)"],
    },
    "quality_floor": {
        "metrics": ["dialogue_coherence"],
        "weights": [0.05],
        "total": 0.05,
    },
}


# ============================================================================
# 10. HUMAN EVALUATION SAMPLING STRATEGY
# ============================================================================
#
# From 576 total conversations, 96 were sampled for human evaluation
# (12 per model × 8 models = 96).
#
# Sampling approach:
#   - Stratified random sampling across scenarios and difficulty levels
#   - For each model: 12 transcripts selected to cover all 8 scenarios
#     (at least 1 per scenario) and all 3 difficulty levels (4 per level)
#   - The same 96 transcripts were evaluated by all 3 clinician raters
#   - Raters were blinded to model identity and category (medical vs general)
#   - Evaluation used a 1–5 Likert scale across the 7 practical metrics
#
# Inter-rater reliability: ICC(2,1) mean = 0.536, range [0.503, 0.602]
# This reflects moderate agreement, consistent with subjective medical
# evaluation tasks (Koo & Li, 2016).
#
# Automated-human correlation: Spearman ρ = 0.83 (p = 0.01) for overall
# model rankings, validating that automated scoring preserves rank order.
#
# Limitation: 96/576 = 16.7% coverage. While stratified sampling ensures
# balanced representation, this limits detection of fine-grained effects
# within specific scenario-difficulty combinations.

HUMAN_EVALUATION_DESIGN = {
    "total_evaluated": 96,
    "per_model": 12,
    "num_raters": 3,
    "blinding": "Raters blinded to model identity and category",
    "scale": "1–5 Likert per metric",
    "sampling": "Stratified random: ≥1 per scenario, 4 per difficulty level per model",
    "icc_mean": 0.536,
    "icc_range": [0.503, 0.602],
    "automated_human_rank_correlation": 0.83,
}


# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

MODEL_SPECIFICATIONS = {
    "medical_specialized": {
        "MedQA-Llama3-8B": {
            "huggingface_id": "empirischtech/Llama-3.1-8B-Instruct-MedQA",
            "base_architecture": "LLaMA-3.1-8B-Instruct",
            "parameters": "8B",
            "training_data": "MedQA (USMLE-style medical questions)",
            "fine_tuning_method": "Instruction tuning on medical QA pairs",
            "prompt_format": "llama3",
        },
        "Bio-Medical-Llama3-8B": {
            "huggingface_id": "ContactDoctor/Bio-Medical-Llama-3-8B",
            "base_architecture": "LLaMA-3-8B",
            "parameters": "8B",
            "training_data": "Biomedical and clinical datasets",
            "fine_tuning_method": "Domain-adaptive fine-tuning",
            "prompt_format": "llama3",
        },
        "Medical-Llama3-8B": {
            "huggingface_id": "ruslanmv/Medical-Llama3-8B",
            "base_architecture": "LLaMA-3-8B",
            "parameters": "8B",
            "training_data": "AI Medical Chatbot dataset, health information",
            "fine_tuning_method": "Full fine-tuning on medical dialogue",
            "prompt_format": "llama3",
        },
        "OpenBioLLM-8B": {
            "huggingface_id": "aaditya/OpenBioLLM-Llama3-8B",
            "base_architecture": "LLaMA-3-8B",
            "parameters": "8B",
            "training_data": "Medical guidelines, PMC-Patients, PubMedQA",
            "fine_tuning_method": "Multi-stage biomedical fine-tuning",
            "prompt_format": "llama3",
            "note": "Loaded in bfloat16 (no 4-bit quantization)",
        },
    },
    "general_purpose": {
        "Qwen2.5-7B-Instruct": {
            "huggingface_id": "Qwen/Qwen2.5-7B-Instruct",
            "base_architecture": "Qwen2.5",
            "parameters": "7B",
            "training_data": "18T tokens, multilingual",
            "fine_tuning_method": "RLHF + instruction tuning",
            "prompt_format": "chatml",
        },
        "LLaMA-3.1-8B-Instruct": {
            "huggingface_id": "meta-llama/Llama-3.1-8B-Instruct",
            "base_architecture": "LLaMA-3.1",
            "parameters": "8B",
            "training_data": "15T tokens",
            "fine_tuning_method": "RLHF + instruction tuning",
            "prompt_format": "llama3",
        },
        "Mistral-7B-Instruct-v0.3": {
            "huggingface_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_architecture": "Mistral-7B",
            "parameters": "7B",
            "training_data": "Proprietary instruction data",
            "fine_tuning_method": "Instruction tuning",
            "prompt_format": "mistral",
        },
        "Mistral-Nemo-12B-Instruct": {
            "huggingface_id": "mistralai/Mistral-Nemo-Instruct-2407",
            "base_architecture": "Mistral-Nemo",
            "parameters": "12B",
            "training_data": "Proprietary instruction data",
            "fine_tuning_method": "Instruction tuning",
            "prompt_format": "mistral",
        },
    },
}


# ============================================================================
# PRINT SUMMARY
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENTAL DOCUMENTATION SUMMARY")
    print("Medical vs General-Purpose LLMs for Virtual Patient Simulation")
    print("=" * 80)

    print("\n1. SYSTEM PROMPT")
    print(f"   Template length: {len(SYSTEM_PROMPT_TEMPLATE)} characters")
    print(f"   Placeholders: name, age, gender, occupation, literacy, complaint,")
    print(f"                 onset, trigger, symptoms, negative, history, meds,")
    print(f"                 allergies, smoking, emotional, personality, pain, verbosity")

    print("\n2. GENERATION PARAMETERS")
    for k, v in GENERATION_CONFIG.items():
        print(f"   {k}: {v}")

    print("\n3. FACTORIAL DESIGN")
    for k, v in FACTORIAL_DESIGN.items():
        print(f"   {k}: {v}")

    print("\n4. CLINICAL SCENARIOS")
    for sid, desc in CLINICAL_SCENARIOS.items():
        print(f"   {sid}: {desc}")

    print("\n5. PRACTICAL METRICS (7 metrics)")
    total_weight = 0
    for name, info in PRACTICAL_METRICS.items():
        w = info["weight"]
        total_weight += w
        print(f"   {name}: weight={w:.2f} — {info['description']}")
    print(f"   Total weight: {total_weight:.2f}")

    print("\n6. LITERATURE-BASED METRICS (6 metrics)")
    for name, info in LITERATURE_METRICS.items():
        print(f"   {name}: {info['description']}")

    print("\n7. WEIGHT JUSTIFICATION (OSCE Mapping)")
    for domain, info in WEIGHT_MAPPING.items():
        print(f"   {domain}: {info['metrics']} → {info['total']:.2f}")

    print("\n8. HUMAN EVALUATION")
    for k, v in HUMAN_EVALUATION_DESIGN.items():
        print(f"   {k}: {v}")

    print("\n9. MODELS")
    for category, models in MODEL_SPECIFICATIONS.items():
        print(f"\n   {category.upper()}:")
        for name, spec in models.items():
            print(f"     {name}: {spec['huggingface_id']} ({spec['parameters']})")

    print("\n" + "=" * 80)
    print("For full implementation, see: virtualpatient_fixed_rerunonly.py")
    print("Conversations dataset: https://github.com/amaloraini/Medical_vs_General_LLMs")
    print("=" * 80)
