import io
import math
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Emoji Emotion Encoder", layout="wide")

NEUTRAL_POINT = {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}


def make_default_lexicon() -> pd.DataFrame:
    data = [
        ["😀", "grinning face", 0.82, 0.72, 0.68, "joy", "contentment", "high positive affect"],
        ["😃", "grinning face with big eyes", 0.86, 0.76, 0.70, "joy", "excitement", "high positive affect"],
        ["😄", "grinning face with smiling eyes", 0.90, 0.74, 0.71, "joy", "contentment", "warm positive affect"],
        ["😁", "beaming face with smiling eyes", 0.88, 0.80, 0.73, "joy", "pride", "strong positive affect"],
        ["😊", "smiling face with smiling eyes", 0.78, 0.55, 0.63, "contentment", "joy", "gentle positivity"],
        ["🙂", "slightly smiling face", 0.55, 0.38, 0.57, "contentment", "joy", "mild positive affect"],
        ["😉", "winking face", 0.52, 0.50, 0.70, "playfulness", "joy", "social confidence"],
        ["😌", "relieved face", 0.50, 0.24, 0.56, "relief", "contentment", "low-arousal positive affect"],
        ["😍", "smiling face with heart-eyes", 0.93, 0.84, 0.75, "love", "joy", "high-approach positivity"],
        ["🥰", "smiling face with hearts", 0.95, 0.78, 0.72, "love", "gratitude", "warm affiliative emotion"],
        ["😘", "face blowing a kiss", 0.88, 0.68, 0.73, "love", "playfulness", "positive social signal"],
        ["🤗", "hugging face", 0.76, 0.58, 0.62, "affection", "gratitude", "warmth and support"],
        ["🤩", "star-struck", 0.91, 0.88, 0.74, "awe", "joy", "intense positive activation"],
        ["🥳", "partying face", 0.89, 0.92, 0.76, "celebration", "joy", "very high arousal positive affect"],
        ["😎", "smiling face with sunglasses", 0.70, 0.48, 0.84, "confidence", "joy", "high dominance positive affect"],
        ["🤔", "thinking face", 0.02, 0.42, 0.62, "contemplation", "uncertainty", "cognitively engaged"],
        ["🫤", "face with diagonal mouth", -0.18, 0.32, 0.38, "ambivalence", "confusion", "mixed affect"],
        ["😐", "neutral face", 0.00, 0.20, 0.50, "neutral", "flat", "benchmark neutral"],
        ["😑", "expressionless face", -0.10, 0.16, 0.44, "flat", "disengagement", "low affect"],
        ["😶", "face without mouth", -0.06, 0.18, 0.40, "withdrawal", "silence", "suppressed response"],
        ["🙃", "upside-down face", 0.08, 0.52, 0.47, "sarcasm", "playfulness", "ambiguous valence"],
        ["😏", "smirking face", 0.18, 0.52, 0.79, "confidence", "sarcasm", "socially dominant ambiguity"],
        ["😬", "grimacing face", -0.42, 0.72, 0.35, "awkwardness", "anxiety", "tense negative affect"],
        ["😅", "grinning face with sweat", 0.18, 0.76, 0.42, "relief", "anxiety", "mixed nervous positivity"],
        ["😂", "face with tears of joy", 0.88, 0.94, 0.69, "joy", "amusement", "intense laughter"],
        ["🤣", "rolling on the floor laughing", 0.92, 0.98, 0.67, "amusement", "joy", "peak laughter"],
        ["😭", "loudly crying face", -0.92, 0.84, 0.16, "sadness", "distress", "intense negative affect"],
        ["😢", "crying face", -0.78, 0.56, 0.22, "sadness", "disappointment", "clear sadness"],
        ["☹️", "frowning face", -0.62, 0.38, 0.28, "sadness", "disappointment", "mild-to-moderate negative affect"],
        ["🙁", "slightly frowning face", -0.46, 0.30, 0.34, "sadness", "concern", "mild negative affect"],
        ["😔", "pensive face", -0.54, 0.28, 0.30, "sadness", "rumination", "low-arousal negative affect"],
        ["😞", "disappointed face", -0.62, 0.40, 0.28, "disappointment", "sadness", "goal frustration"],
        ["😕", "confused face", -0.30, 0.42, 0.34, "confusion", "concern", "uncertain negative affect"],
        ["😟", "worried face", -0.58, 0.60, 0.24, "worry", "anxiety", "anticipatory threat"],
        ["😰", "anxious face with sweat", -0.76, 0.84, 0.18, "anxiety", "fear", "high-arousal distress"],
        ["😨", "fearful face", -0.82, 0.86, 0.14, "fear", "anxiety", "acute threat"],
        ["😱", "face screaming in fear", -0.90, 0.96, 0.10, "fear", "shock", "extreme high-arousal negative affect"],
        ["😳", "flushed face", -0.18, 0.72, 0.30, "embarrassment", "surprise", "social exposure"],
        ["🥺", "pleading face", -0.24, 0.44, 0.16, "need", "sadness", "low dominance appeal"],
        ["😤", "face with steam from nose", -0.24, 0.80, 0.72, "frustration", "anger", "activated frustration"],
        ["😠", "angry face", -0.78, 0.82, 0.78, "anger", "frustration", "high-dominance negative affect"],
        ["😡", "pouting face", -0.88, 0.90, 0.84, "anger", "rage", "extreme angry affect"],
        ["🤬", "face with symbols on mouth", -0.92, 0.96, 0.86, "rage", "anger", "maximal hostile activation"],
        ["😒", "unamused face", -0.52, 0.32, 0.54, "irritation", "disapproval", "low-arousal negative judgment"],
        ["🙄", "face with rolling eyes", -0.40, 0.40, 0.62, "disdain", "irritation", "social dismissal"],
        ["😴", "sleeping face", 0.04, 0.04, 0.46, "fatigue", "calm", "very low arousal"],
        ["🥱", "yawning face", -0.06, 0.06, 0.42, "fatigue", "boredom", "very low arousal"],
        ["🤯", "exploding head", -0.10, 0.94, 0.34, "shock", "surprise", "cognitive overload"],
        ["😮", "face with open mouth", 0.02, 0.74, 0.40, "surprise", "awe", "high arousal, unclear valence"],
        ["😲", "astonished face", -0.04, 0.82, 0.34, "surprise", "shock", "high arousal ambiguity"],
        ["❤️", "red heart", 0.96, 0.62, 0.66, "love", "affection", "strong positive social bond"],
        ["💔", "broken heart", -0.90, 0.58, 0.20, "loss", "sadness", "social pain"],
        ["👍", "thumbs up", 0.68, 0.46, 0.68, "approval", "confidence", "positive endorsement"],
        ["👎", "thumbs down", -0.66, 0.44, 0.58, "disapproval", "anger", "negative evaluation"],
        ["👏", "clapping hands", 0.74, 0.70, 0.64, "approval", "celebration", "social reinforcement"],
        ["🙏", "folded hands", 0.52, 0.28, 0.34, "gratitude", "hope", "low-arousal affiliative signal"],
        ["🔥", "fire", 0.26, 0.92, 0.72, "excitement", "intensity", "high activation marker"],
        ["💀", "skull", -0.22, 0.70, 0.40, "dark humor", "shock", "context-dependent"],
        ["🎉", "party popper", 0.86, 0.88, 0.66, "celebration", "joy", "positive event signal"],
        ["✨", "sparkles", 0.62, 0.52, 0.60, "optimism", "awe", "light positive accent"],
    ]
    return pd.DataFrame(
        data,
        columns=[
            "emoji",
            "label",
            "valence",
            "arousal",
            "dominance",
            "primary_family",
            "secondary_family",
            "notes",
        ],
    )


def normalize_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "emoji",
        "label",
        "valence",
        "arousal",
        "dominance",
        "primary_family",
        "secondary_family",
        "notes",
    ]

    out = df.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = "" if col in {"emoji", "label", "primary_family", "secondary_family", "notes"} else np.nan

    out = out[expected]
    out["emoji"] = out["emoji"].astype(str).str.strip()
    out = out[out["emoji"] != ""]

    for col in ["valence", "arousal", "dominance"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["valence", "arousal", "dominance"])
    out = out.drop_duplicates(subset=["emoji"], keep="last").reset_index(drop=True)

    out["valence"] = out["valence"].clip(-1, 1)
    out["arousal"] = out["arousal"].clip(0, 1)
    out["dominance"] = out["dominance"].clip(0, 1)
    out["intensity"] = np.sqrt(
        (out["valence"] - NEUTRAL_POINT["valence"]) ** 2
        + (out["arousal"] - NEUTRAL_POINT["arousal"]) ** 2
        + (out["dominance"] - NEUTRAL_POINT["dominance"]) ** 2
    )
    return out


def build_emoji_pattern(emojis: List[str]):
    usable = [e for e in emojis if isinstance(e, str) and e]
    if not usable:
        return None
    usable = sorted(set(usable), key=len, reverse=True)
    return re.compile("|".join(re.escape(e) for e in usable))


def extract_mapped_emojis(text: str, pattern) -> List[str]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    text = str(text)
    if not text.strip() or pattern is None:
        return []
    return pattern.findall(text)


def summarize_emojis(emojis: List[str], lexicon_indexed: pd.DataFrame, agg_method: str) -> Dict[str, object]:
    hits = [e for e in emojis if e in lexicon_indexed.index]
    if not hits:
        return {
            "emoji_count": 0,
            "emoji_detected": "",
            "valence": np.nan,
            "arousal": np.nan,
            "dominance": np.nan,
            "intensity": np.nan,
            "primary_family": "",
            "secondary_family": "",
        }

    rows = lexicon_indexed.loc[hits].copy()
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T

    if agg_method == "sum":
        valence = rows["valence"].sum()
        arousal = rows["arousal"].sum()
        dominance = rows["dominance"].sum()
        intensity = rows["intensity"].sum()
    else:
        valence = rows["valence"].mean()
        arousal = rows["arousal"].mean()
        dominance = rows["dominance"].mean()
        intensity = rows["intensity"].mean()

    primary_mode = rows["primary_family"].mode(dropna=True)
    secondary_mode = rows["secondary_family"].mode(dropna=True)

    return {
        "emoji_count": len(hits),
        "emoji_detected": " ".join(hits),
        "valence": round(float(valence), 4),
        "arousal": round(float(arousal), 4),
        "dominance": round(float(dominance), 4),
        "intensity": round(float(intensity), 4),
        "primary_family": "" if primary_mode.empty else str(primary_mode.iloc[0]),
        "secondary_family": "" if secondary_mode.empty else str(secondary_mode.iloc[0]),
    }


def make_family_weights(emojis: List[str], lexicon_indexed: pd.DataFrame, families: List[str]) -> Dict[str, float]:
    hits = [e for e in emojis if e in lexicon_indexed.index]
    out = {f"family_weight_{slugify(f)}": 0.0 for f in families}
    if not hits:
        return out
    rows = lexicon_indexed.loc[hits]
    if isinstance(rows, pd.Series):
        rows = rows.to_frame().T
    counts = rows["primary_family"].value_counts(normalize=True)
    for family in families:
        out[f"family_weight_{slugify(family)}"] = round(float(counts.get(family, 0.0)), 4)
    return out


def slugify(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def transform_dataframe(
    df: pd.DataFrame,
    source_col: str,
    lexicon: pd.DataFrame,
    agg_method: str,
    include_family_weights: bool,
) -> pd.DataFrame:
    lexicon_indexed = lexicon.set_index("emoji", drop=False)
    pattern = build_emoji_pattern(lexicon["emoji"].tolist())
    families = sorted([f for f in lexicon["primary_family"].dropna().astype(str).unique().tolist() if f.strip()])

    transformed = df.copy()
    extracted = transformed[source_col].apply(lambda x: extract_mapped_emojis(x, pattern))

    summaries = extracted.apply(lambda x: summarize_emojis(x, lexicon_indexed, agg_method))
    summary_df = pd.DataFrame(list(summaries))

    transformed["emoji_detected"] = summary_df["emoji_detected"]
    transformed["emoji_count"] = summary_df["emoji_count"]
    transformed["emoji_valence"] = summary_df["valence"]
    transformed["emoji_arousal"] = summary_df["arousal"]
    transformed["emoji_dominance"] = summary_df["dominance"]
    transformed["emoji_intensity"] = summary_df["intensity"]
    transformed["emoji_primary_family"] = summary_df["primary_family"]
    transformed["emoji_secondary_family"] = summary_df["secondary_family"]

    if include_family_weights:
        family_df = pd.DataFrame(list(extracted.apply(lambda x: make_family_weights(x, lexicon_indexed, families))))
        transformed = pd.concat([transformed, family_df], axis=1)

    return transformed


def dataframe_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().split(".")[-1]
    if suffix == "csv":
        return pd.read_csv(uploaded_file)
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


st.title("😀 Emoji Emotion Encoder")
st.write(
    "Convert emojis into continuous emotion variables for quantitative analysis using an editable VAD-style lexicon "
    "(valence, arousal, dominance), plus derived intensity and emotion-family outputs."
)

with st.expander("Measurement note", expanded=False):
    st.write(
        "This app uses a dimensional model rather than treating emojis as purely categorical labels. The default lexicon is a practical starter set, "
        "not a substitute for normative validation in your own study population. For publication-grade psychometric work, you should revise or norm the emoji scores in your target sample."
    )
    st.write(
        "Scales used here: valence = -1 to +1, arousal = 0 to 1, dominance = 0 to 1. Intensity is the Euclidean distance from a neutral point of (0, 0.5, 0.5)."
    )

if "lexicon_df" not in st.session_state:
    st.session_state.lexicon_df = make_default_lexicon()

with st.sidebar:
    st.header("Settings")
    agg_method = st.selectbox(
        "How should multiple emojis in the same cell be combined?",
        options=["mean", "sum"],
        index=0,
        help="Mean is usually the best default for psychometric analysis; sum may be useful when emoji frequency itself carries meaning.",
    )
    include_family_weights = st.checkbox(
        "Add continuous family-weight columns",
        value=True,
        help="Creates columns such as family_weight_joy, family_weight_sadness, etc., based on the proportion of emojis in each row belonging to each primary family.",
    )

    st.divider()
    st.subheader("Lexicon import")
    lexicon_upload = st.file_uploader(
        "Upload custom lexicon (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        help="Expected columns: emoji, label, valence, arousal, dominance, primary_family, secondary_family, notes.",
        key="lexicon_upload",
    )
    if lexicon_upload is not None:
        try:
            imported_lexicon = read_uploaded_table(lexicon_upload)
            st.session_state.lexicon_df = imported_lexicon.copy()
            st.success("Custom lexicon loaded into the editor below.")
        except Exception as e:
            st.error(f"Could not read lexicon file: {e}")

    if st.button("Reset to default lexicon"):
        st.session_state.lexicon_df = make_default_lexicon()
        st.success("Default lexicon restored.")

lexicon_header_col1, lexicon_header_col2 = st.columns([3, 1])
with lexicon_header_col1:
    st.subheader("Editable emoji lexicon")
with lexicon_header_col2:
    template_bytes = dataframe_to_csv_download(make_default_lexicon())
    st.download_button(
        "Download template",
        data=template_bytes,
        file_name="emoji_lexicon_template.csv",
        mime="text/csv",
    )

edited_lexicon = st.data_editor(
    st.session_state.lexicon_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "emoji": st.column_config.TextColumn("emoji", help="The exact emoji string to match."),
        "label": st.column_config.TextColumn("label"),
        "valence": st.column_config.NumberColumn("valence", help="-1 = strongly negative, +1 = strongly positive", min_value=-1.0, max_value=1.0, step=0.01),
        "arousal": st.column_config.NumberColumn("arousal", help="0 = very calm/inactive, 1 = highly activated", min_value=0.0, max_value=1.0, step=0.01),
        "dominance": st.column_config.NumberColumn("dominance", help="0 = low control/power, 1 = high control/power", min_value=0.0, max_value=1.0, step=0.01),
        "primary_family": st.column_config.TextColumn("primary_family"),
        "secondary_family": st.column_config.TextColumn("secondary_family"),
        "notes": st.column_config.TextColumn("notes"),
    },
    key="lexicon_editor",
)

lexicon = normalize_lexicon(edited_lexicon)
st.caption(f"Active lexicon size: {len(lexicon)} emoji entries")

if lexicon.empty:
    st.error("Your lexicon is empty after validation. Please add at least one emoji row with numeric valence, arousal, and dominance values.")
    st.stop()

st.divider()

lookup_col, details_col = st.columns([1, 1])
with lookup_col:
    st.subheader("Quick lookup / string conversion")
    quick_text = st.text_area(
        "Enter one emoji or a text string containing emojis",
        value="I felt 😟 before the exam but later became 😌 and 😀",
        height=120,
    )
    if st.button("Convert text", use_container_width=True):
        pattern = build_emoji_pattern(lexicon["emoji"].tolist())
        extracted = extract_mapped_emojis(quick_text, pattern)
        summary = summarize_emojis(extracted, lexicon.set_index("emoji", drop=False), agg_method)
        family_weights = make_family_weights(
            extracted,
            lexicon.set_index("emoji", drop=False),
            sorted([f for f in lexicon["primary_family"].dropna().astype(str).unique().tolist() if f.strip()]),
        )

        st.session_state.quick_extracted = extracted
        st.session_state.quick_summary = summary
        st.session_state.quick_family_weights = family_weights

with details_col:
    st.subheader("Converted outputs")
    if "quick_summary" in st.session_state:
        st.write("Detected emojis:", st.session_state.quick_summary["emoji_detected"] or "None")
        metric_df = pd.DataFrame([
            {
                "emoji_count": st.session_state.quick_summary["emoji_count"],
                "valence": st.session_state.quick_summary["valence"],
                "arousal": st.session_state.quick_summary["arousal"],
                "dominance": st.session_state.quick_summary["dominance"],
                "intensity": st.session_state.quick_summary["intensity"],
                "primary_family": st.session_state.quick_summary["primary_family"],
                "secondary_family": st.session_state.quick_summary["secondary_family"],
            }
        ])
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        if st.session_state.quick_extracted:
            hit_df = lexicon[lexicon["emoji"].isin(st.session_state.quick_extracted)].copy()
            hit_df = hit_df[["emoji", "label", "valence", "arousal", "dominance", "intensity", "primary_family", "secondary_family"]]
            st.write("Matched emoji rows")
            st.dataframe(hit_df, use_container_width=True, hide_index=True)

        if include_family_weights:
            family_weight_df = pd.DataFrame([st.session_state.quick_family_weights])
            st.write("Family weights")
            st.dataframe(family_weight_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the quick conversion to preview how emoji strings will be encoded.")

st.divider()
st.subheader("Batch conversion for analysis datasets")
uploaded = st.file_uploader(
    "Upload a CSV or Excel dataset",
    type=["csv", "xlsx", "xls"],
    help="Upload a dataset containing a column with emojis or free text that includes emojis.",
    key="data_upload",
)

if uploaded is not None:
    try:
        source_df = read_uploaded_table(uploaded)
        if source_df.empty:
            st.warning("The uploaded dataset is empty.")
        else:
            st.write("Preview of uploaded data")
            st.dataframe(source_df.head(10), use_container_width=True)

            emoji_candidate_cols = source_df.columns.tolist()
            selected_col = st.selectbox(
                "Which column should be converted?",
                options=emoji_candidate_cols,
                help="The app will search that column for any emojis included in the lexicon.",
            )

            transformed_df = transform_dataframe(
                df=source_df,
                source_col=selected_col,
                lexicon=lexicon,
                agg_method=agg_method,
                include_family_weights=include_family_weights,
            )

            st.write("Transformed data preview")
            st.dataframe(transformed_df.head(25), use_container_width=True)

            nonmissing = transformed_df["emoji_count"].fillna(0).gt(0).sum()
            st.caption(f"Rows with at least one matched emoji: {nonmissing} of {len(transformed_df)}")

            download_bytes = dataframe_to_csv_download(transformed_df)
            st.download_button(
                "Download transformed CSV",
                data=download_bytes,
                file_name="emoji_encoded_output.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Could not process the uploaded dataset: {e}")
else:
    st.info("Upload a CSV or Excel dataset to create row-level emotion variables for downstream statistical analysis.")

st.divider()
st.subheader("How to use these outputs in analysis")
st.write(
    "The most directly usable continuous predictors are emoji_valence, emoji_arousal, emoji_dominance, and emoji_intensity. "
    "If you enable family weights, those columns can be used as additional continuous indicators or descriptive features."
)
st.write(
    "Typical use cases include linear regression, mixed-effects models, scale development, convergent validity checks, and sensitivity analyses comparing dimensional coding against simpler categorical encodings."
)
