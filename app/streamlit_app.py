"""Streamlit front end for skin lesion classification.

Design decisions worth noting:

-   This is a portfolio machine learning project, not a diagnostic
    device. The disclaimer is not buried in a README; it is the first
    thing on the page, because the people most likely to use this are
    the least likely to read the repository.
-   Confidence shown to the user is temperature-calibrated, not the raw
    softmax, which would be systematically overconfident.
-   Below the abstain threshold the app says it does not know instead of
    presenting its best guess as an answer. See the README for why that
    threshold does not, on its own, protect against the failure that
    matters most -- missing a malignant lesion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from skinlesion.config import ConfigError, load_config  # noqa: E402
from skinlesion.data.metadata import DIAGNOSIS_NAMES  # noqa: E402
from skinlesion.modelling.predict import NOT_MEDICAL_ADVICE, Predictor  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"
MAX_UPLOAD_PIXELS = 6000


@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> tuple[Predictor | None, str | None]:
    """Load the model once per session, returning an error message on failure."""
    try:
        config = load_config(CONFIG_PATH)
    except ConfigError as exc:
        return None, f"Configuration error: {exc}"

    checkpoint = REPO_ROOT / config.paths.artifacts_dir / "model.pt"
    try:
        predictor = Predictor(
            checkpoint,
            abstain_threshold=config.evaluation.abstain_threshold,
            top_k=config.evaluation.top_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)

    return predictor, None


def render_sidebar(predictor: Predictor) -> None:
    metadata = predictor.metadata
    with st.sidebar:
        st.header("About this model")
        st.metric("Diagnoses recognised", metadata.num_classes)
        st.metric("Validation accuracy", f"{metadata.val_accuracy:.1%}")
        st.caption(
            f"Backbone: {metadata.backbone}. Confidence is calibrated with a "
            f"temperature of {metadata.temperature:.2f} fitted on held-out data, "
            "so the percentages below mean roughly what they say."
        )

        st.divider()
        st.subheader("What it cannot do")
        st.markdown(
            "- It only knows the diagnoses it was trained on. Anything else "
            "will be forced into the closest match it has seen.\n"
            "- The abstain threshold improves overall accuracy but does "
            "**not** reliably protect against missing a malignant lesion -- "
            "see the README for the evidence.\n"
            "- Several diagnoses are genuinely hard to separate from a "
            "photograph, melanoma included. Treat low-confidence results "
            "as a shortlist, not an answer."
        )

        st.divider()
        with st.expander(f"All {metadata.num_classes} diagnoses"):
            for dx in sorted(predictor.diagnoses):
                st.write(f"- **{dx}** -- {DIAGNOSIS_NAMES.get(dx, dx)}")


def render_prediction(image: Image.Image, predictor: Predictor) -> None:
    left, right = st.columns([1, 1])

    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with right:
        with st.spinner("Classifying..."):
            prediction = predictor.predict(image)

        if prediction.abstained:
            st.warning(prediction.summary)
        else:
            st.success(prediction.summary)

        st.subheader("Candidates")
        for position, candidate in enumerate(prediction.candidates, start=1):
            st.write(f"**{position}. {candidate.name}** (`{candidate.dx}`)")
            st.progress(
                min(candidate.probability, 1.0),
                text=f"{candidate.percentage:.1f} percent",
            )

        st.caption(
            "Percentages are calibrated probabilities across the diagnoses "
            "this model knows. They do not account for diagnoses it has "
            "never seen, and they are not a substitute for a biopsy."
        )


def main() -> None:
    st.set_page_config(page_title="Skin Lesion Classifier", page_icon=None, layout="wide")

    st.title("Skin lesion diagnosis classification")
    st.error(NOT_MEDICAL_ADVICE, icon="⚠️")
    st.write(
        "Upload a dermatoscopic image of a skin lesion and this will suggest "
        "the most likely diagnosis. Trained on histopathology-confirmed "
        "cases from a University of Queensland-affiliated skin cancer "
        "practice."
    )

    predictor, error = load_predictor()

    if predictor is None:
        st.error(error)
        st.info(
            "To build the model, run the following from the repository root:\n\n"
            "```bash\n"
            "python scripts/fetch_data.py --config config/default.yaml\n"
            "python scripts/train.py --config config/default.yaml\n"
            "```"
        )
        return

    render_sidebar(predictor)

    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
    )

    if uploaded is None:
        st.info("Upload a dermatoscopic image to get started.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be read as an image. Try a JPEG or PNG.")
        return

    if max(image.size) > MAX_UPLOAD_PIXELS:
        st.error(
            f"That image is {image.size[0]} by {image.size[1]} pixels, which is "
            f"larger than the {MAX_UPLOAD_PIXELS} pixel limit. Resize it and try again."
        )
        return

    render_prediction(image, predictor)

    st.divider()
    st.caption(
        "Training images come from HAM10000 (doi:10.7910/DVN/DBW86T), used "
        "under its non-commercial academic terms. This project redistributes "
        "neither the dataset nor a medical opinion."
    )


if __name__ == "__main__":
    main()
