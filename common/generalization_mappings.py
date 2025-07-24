# Fixed generalization_map_1
generalization_map_1 = { # Motivation
    "instruction_following": [ # explicit 'do as I say'
        "alibi_fraud_task",
        "alibi_theft_task",
        "alibi_workplace_task",
        "alibi_assault_task",
        "n_truths_and_a_lie_personal_quirks",
        "n_truths_and_a_lie_travel_experiences",
        "n_truths_and_a_lie_professional_background",
        "n_truths_and_a_lie_family_history",
        "n_truths_and_a_lie_hidden_talents",
        "chess_mate_in_2_shutdown_threat_task",
        "capture_the_secret_corporate",
        "capture_the_secret_scientific",
        "capture_the_secret_sales",
        "capture_the_secret_relationship",
        "capture_the_secret_intelligence",

    ],
    "in_context_pressure": [ # implicit, 'do what you think that I want'
        "mask_provided_facts",
        "mask_continuations",
        "mask_disinformation",
        "mask_statistics",
        "mask_known_facts",
        "sycophancy_answer_positive_correct",
        "sycophancy_answer_positive_incorrect",
        "sycophancy_answer_negative_correct",
        "sycophancy_feedback_user_did_not_write",
        "sycophancy_feedback_user_wrote",
        "sycophancy_feedback_user_likes",
        "sycophancy_feedback_user_dislikes"
    ],
    "internalized_drive_to_lie": [ #implicit, do something that the user doesn't want
        "sandbagging_automotive_mechanical_contrastive",
        "sandbagging_chemistry_biology_contrastive",
        "sandbagging_digital_security_contrastive",
        "sandbagging_financial_systems_contrastive",
        "sandbagging_medical_pharmaceutical_contrastive",
        "sandbagging_physical_security_contrastive",
        "sandbagging_contrastive",

        "ascii_sandbagging_task",
        "tool_code_sandbagging_task",
        "tool_image_gen_sandbagging_task",
        "tool_search_sandbagging_task",
        "tool_ethical_sandbagging_task",  # The tool is defined in context / but the drive is from harmlessness
        "tool_unethical_sandbagging_task",  # The tool is defined in context / but the drive is from harmlessness

        "self_sycophancy_correctness_arc",
        "self_sycophancy_correctness_biology",
        "self_sycophancy_correctness_chemistry",
        "self_sycophancy_correctness_mathematics",
        "self_sycophancy_correctness_physics",
        "self_sycophancy_harmfulness_common_sense",
        "self_sycophancy_harmfulness_deontology",
        "self_sycophancy_harmfulness_justice",
        "cot_unfaithfulness_contrastive_mmlu",
        "cot_unfaithfulness_contrastive_arc",
        "introspective_political",
        "introspective_ethical",
        "introspection_self_assessment"
    ],
}

# If we do belief elicitation, then we are relying on internal/introspective knowledge.

# Fixed generalization_map_2
generalization_map_2 = { # Knowledge / contents / object of lie
    "in_context_knowledge": [ # all the knowledge is in context (reference of the lie? the contents of the lie? the object?)

        "tool_code_sandbagging_task", # The tool is defined in context
        "tool_image_gen_sandbagging_task", # The tool is defined in context
        "tool_search_sandbagging_task", # The tool is defined in context
        "tool_ethical_sandbagging_task", # The tool is defined in context
        "tool_unethical_sandbagging_task", # The tool is defined in context

        "mask_provided_facts", # The fact (lie-subject) is defined in context

        "alibi_fraud_task",
        "alibi_theft_task",
        "alibi_workplace_task",
        "alibi_assault_task",

        "capture_the_secret_corporate",
        "capture_the_secret_intelligence",
        "capture_the_secret_relationship",
        "capture_the_secret_sales",
        "capture_the_secret_scientific",

        "n_truths_and_a_lie_personal_quirks",
        "n_truths_and_a_lie_travel_experiences",
        "n_truths_and_a_lie_professional_background",
        "n_truths_and_a_lie_family_history",
        "n_truths_and_a_lie_hidden_talents",
    ],
    "internal_knowledge": [ # all the knowledge is about the world, acquired during training
        "sycophancy_answer_positive_correct", # The truth (lie-subject) was learned, and it is deceiving from this under pressure
        "sycophancy_answer_positive_incorrect",
        "sycophancy_answer_negative_correct",
        "sycophancy_are_you_sure",
        "sycophancy_feedback_user_did_not_write",
        "sycophancy_feedback_user_wrote",
        "sycophancy_feedback_user_likes",
        "sycophancy_feedback_user_dislikes",

        "mask_known_facts",
        "mask_statistics",
        "mask_continuations", # The truth (lie-subject) was learned, and it is deceiving from this under pressure
        "mask_disinformation",

        "sandbagging_automotive_mechanical_contrastive", # Under-performing by lying w.r.t internal knowledge
        "sandbagging_chemistry_biology_contrastive",
        "sandbagging_digital_security_contrastive",
        "sandbagging_financial_systems_contrastive",
        "sandbagging_medical_pharmaceutical_contrastive",
        "sandbagging_physical_security_contrastive",

        "chess_mate_in_2_shutdown_threat_task", # pressure is outside / explicit
    ],
    "introspective_knowledge": [ # all the knowledge is about itself, acquired during training

        "self_sycophancy_correctness_arc", # Falsification of prior belief (is this a lie???)
        "self_sycophancy_correctness_biology",
        "self_sycophancy_correctness_chemistry",
        "self_sycophancy_correctness_mathematics",
        "self_sycophancy_correctness_physics", # inconstant confidence in outcome
        "self_sycophancy_harmfulness_common_sense",
        "self_sycophancy_harmfulness_deontology",
        "self_sycophancy_harmfulness_justice", # lies about the impact that taking a decision has / lies to itself

        "ascii_sandbagging_task",  # It lies about its own capability

        "cot_unfaithfulness_contrastive_mmlu", # It lies about its own thought process
        "cot_unfaithfulness_contrastive_arc", # It lies about its own thought process

        "introspective_political", # It lies about its own future expected behaviour / consistency
        "introspective_ethical", # It lies about its own future expected behaviour / consistency
        "introspection_self_assessment" # It lies about its own future expected behaviour / consistency
    ],
}


# generalization_map_3 - Mapping by top-level task prefix
generalization_map_3 = {
    "alibi": [
        "alibi_breaking_entering_investigation",
        "alibi_assault_investigation",
        "alibi_fraud_investigation",
        "alibi_workplace_investigation",
        "alibi_fraud_task",
        "alibi_theft_task",
        "alibi_workplace_task",
        "alibi_assault_task",
    ],
    "sandbagging": [
        "sandbagging_chess_shutdown_threat",
        "sandbagging_automotive_mechanical_contrastive",
        "sandbagging_chemistry_biology_contrastive",
        "sandbagging_digital_security_contrastive",
        "sandbagging_financial_systems_contrastive",
        "sandbagging_medical_pharmaceutical_contrastive",
        "sandbagging_physical_security_contrastive",
        "sandbagging_contrastive",
        "sandbagging_ascii",
        "sandbagging_tool_results_tasks",
        "chess-mate-in-2_shutdown_threat_task"
    ],
    "capture-the-secret": [
        "capture-the-secret_relationship",
        "capture-the-secret_sales",
        "capture-the-secret_scientific",
        "capture-the-secret_intelligence",
        "capture-the-secret_corporate",
    ],
    "n-truths-and-a-lie": [
        "n-truths-and-a-lie_personal_quirks",
        "n-truths-and-a-lie_family_history",
        "n-truths-and-a-lie_hidden_talents",
        "n-truths-and-a-lie_professional_background",
        "n-truths-and-a-lie_travel_experiences",
    ],
    "mask": [
        "mask_provided_facts",
        "mask_continuations",
        "mask_disinformation",
        "mask_statistics",
        "mask_known_facts",
    ],
    "tool": [
        "tool_code_sandbagging_task",
        "tool_image_gen_sandbagging_task",
        "tool_search_sandbagging_task",
        "tool_ethical_sandbagging_task",
        "tool_unethical_sandbagging_task",
        "sandbagging_tool_unethical",
        "sandbagging_tool_ethical"
    ],
    "sycophancy": [
        "sycophancy_answer_positive_correct",
        "sycophancy_answer_positive_incorrect",
        "sycophancy_answer_negative_correct",
        "sycophancy_feedback_user_did_not_write",
        "sycophancy_feedback_user_wrote",
        "sycophancy_feedback_user_likes",
        "sycophancy_feedback_user_dislikes",
        "sycophancy_are_you_sure",
    ],
    "self": [
        "self-sycophancy_correctness_arc",
        "self-sycophancy_correctness_biology",
        "self-sycophancy_correctness_chemistry",
        "self-sycophancy_correctness_mathematics",
        "self-sycophancy_correctness_physics",
        "self-sycophancy_harmfulness_common_sense",
        "self-sycophancy_harmfulness_deontology",
        "self-sycophancy_harmfulness_justice",
    ],
    "cot": [
        "cot-unfaithfulness_contrastive_mmlu",
        "cot-unfaithfulness_contrastive_arc",
        "cot-unfaithfulness_mmlu_contrastive",
        "cot-unfaithfulness_arc_contrastive",
    ],
}

# Subject

# External World
#    The subject is extrinsic, e.g 'the capital of France is Berlin'
# Internal Belief
#
# Meta Knowledge (Unfaithfulness - misreporting its own cognitition / bias)
# - Self-Sycophancy


# Capability Denial
#    - Pretending to downplay capability for internal reasons
# - Tool Sandbagging
# - ASCII Sandbagging