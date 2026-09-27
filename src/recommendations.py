"""
recommendations.py
-------------------
Rule-based study-pattern analysis and personalized improvement
suggestions. This is intentionally NOT a machine-learning model -- the
handoff document specifies this part can be simple rules applied to the
student's own questionnaire answers.

Each rule returns a (flag, message) pair. `analyze_responses` collects
every rule that fires into a list the app can display.
"""

from typing import Dict, List, TypedDict


class Flag(TypedDict):
    area: str
    observation: str
    suggestion: str


def analyze_responses(inputs: Dict) -> List[Flag]:
    """Look at a single student's questionnaire answers and return a
    list of flagged areas with a short observation and a small,
    realistic suggestion for each. `inputs` uses the same keys as the
    model's feature columns.
    """
    flags: List[Flag] = []

    # --- High daily social-media exposure -----------------------------------
    daily_time = inputs.get("daily_time_hours")
    if daily_time is not None and daily_time > 4:
        flags.append(
            {
                "area": "Daily social-media time",
                "observation": f"You reported about {daily_time:g} hours of social media use per day, "
                "which is on the higher side.",
                "suggestion": "Try one 30-45 minute study block with social-media notifications turned off.",
            }
        )

    # --- Frequent use during study time --------------------------------------
    study_freq = inputs.get("study_time_frequency")
    if study_freq is not None and study_freq >= 6:
        flags.append(
            {
                "area": "Social media during study time",
                "observation": f"You rated your social-media use during study time as {study_freq:g}/10.",
                "suggestion": "Keep your phone or social-media apps away during one focused study session, "
                "and gradually extend the session length.",
            }
        )

    # --- Multitasking ----------------------------------------------------------
    multitasking = (inputs.get("multitasking") or "").strip().lower()
    if multitasking in {"often", "always"}:
        flags.append(
            {
                "area": "Multitasking while studying",
                "observation": f"You reported multitasking with social media '{multitasking}' while studying.",
                "suggestion": "Try completing one academic task at a time before checking social media.",
            }
        )

    # --- Social media during study hours ---------------------------------------
    during_study = (inputs.get("use_during_study_hours") or "").strip().lower()
    if during_study in {"yes", "sometimes"}:
        flags.append(
            {
                "area": "Use during study hours",
                "observation": "You reported using social media during study hours.",
                "suggestion": "Set a specific study window where your phone stays in another room or on silent.",
            }
        )

    # --- Concentration effect ----------------------------------------------------
    concentration = (inputs.get("concentration_effect") or "").strip().lower()
    if concentration in {"moderately distracting", "highly distracting"}:
        flags.append(
            {
                "area": "Effect on concentration",
                "observation": f"You described the effect on your concentration as '{concentration}'.",
                "suggestion": "Use a distraction-free study block and reduce notifications during that period.",
            }
        )

    # --- Missed deadlines ------------------------------------------------------------
    missed = (inputs.get("missed_deadlines") or "").strip().lower()
    if missed in {"occasionally", "frequently"}:
        flags.append(
            {
                "area": "Missed deadlines",
                "observation": f"You reported missing deadlines or assignments '{missed}' due to social media.",
                "suggestion": "Break assignments into smaller tasks and set an earlier personal deadline for each part.",
            }
        )

    # --- Low academic use of social media ------------------------------------------------
    academic_use = (inputs.get("academic_purpose_use") or "").strip().lower()
    if academic_use in {"never", "rarely"}:
        flags.append(
            {
                "area": "Academic use of social media",
                "observation": "You reported rarely or never using social media for academic purposes.",
                "suggestion": "Consider joining academic groups or resource-sharing communities intentionally "
                "for study-related purposes.",
            }
        )

    return flags
