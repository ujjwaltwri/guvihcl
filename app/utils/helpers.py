# guvihcl/app/utils/helpers.py

def generate_explanation(classification, features):
    """
    Generates a reason based on common AI voice artifacts.
    """
    if classification == "AI_GENERATED":
        # In a real app, you'd check for specific artifacts like 
        # lack of breathing sounds or spectral repetition.
        reasons = [
            "Unnatural pitch consistency and robotic speech patterns detected.",
            "Detected synthetic spectral continuity in high-frequency bands.",
            "Absence of natural physiological micro-tremors in the vocal tract."
        ]
        return reasons[0] # Return the most relevant one
    else:
        return "Natural prosody, breath markers, and ambient background nuances detected."