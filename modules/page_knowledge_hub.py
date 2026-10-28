"""AgriVision AI Knowledge Hub.

Standalone Streamlit page.  No trained models or network connectivity required.
Integration: app.py already calls ``show()``.
"""

from __future__ import annotations

import re
import streamlit as st


# These definitions are educational summaries, not field-specific diagnoses.
TERMS = [
    ("NDVI", "Plant health", "Normalized Difference Vegetation Index", "Vegetation index calculated from near-infrared and red reflectance: (NIR − Red)/(NIR + Red). It measures relative canopy greenness, not disease directly.", "NDVI can vary with crop, growth stage, soil background, lighting and sensor. A single universal 'healthy' range is not reliable.", "Soil Analysis / Dashboard"),
    ("Chlorophyll / SPAD", "Plant health", "Leaf pigment and meter readings", "Chlorophyll absorbs light for photosynthesis. SPAD meters provide a relative optical reading, not an exact chlorophyll concentration.", "Interpret with crop- and growth-stage-specific reference data; SPAD alone cannot diagnose nitrogen deficiency.", "Crop Health"),
    ("Leaf area index (LAI)", "Plant health", "Canopy leaf area", "Total one-sided leaf area relative to ground area. It helps describe canopy development and light interception.", "Higher is not always better: very dense canopies can increase shading and humidity.", "Yield Prediction"),
    ("Chlorosis", "Plant health", "Yellowing of green tissues", "Yellowing can result from nutrient deficiency, water stress, root damage, infection or natural senescence.", "Observe which leaves yellow first and check moisture, root health and soil tests before treating.", "Smart Vision / Nutrient Advisor"),
    ("Necrosis", "Plant health", "Dead plant tissue", "Brown or black dead areas can occur after disease, nutrient imbalance, drought, chemical injury or temperature stress.", "The image appearance alone is not enough to identify its cause.", "Smart Vision"),
    ("Nitrogen (N)", "Nutrients", "Vegetative growth and proteins", "Nitrogen supports chlorophyll synthesis, enzymes and leaf growth. Deficiency often affects older leaves first.", "Excess nitrogen can cause excessive foliage and increase environmental losses; use a soil test and local recommendations.", "Nutrient Advisor"),
    ("Phosphorus (P)", "Nutrients", "Energy transfer and roots", "Phosphorus is needed for ATP, establishment and growth. Deficiency can stunt plants; purple coloration occurs in some crops.", "Cold soil and pH can restrict availability. Leaf color alone cannot confirm deficiency.", "Nutrient Advisor"),
    ("Potassium (K)", "Nutrients", "Water regulation and stress response", "Potassium supports stomatal regulation, enzymes and transport of sugars. Some deficiencies cause leaf-edge scorch.", "Scorch also occurs with salinity and drought. Verify with soil or tissue testing.", "Nutrient Advisor"),
    ("Soil pH", "Soil & water", "Acidity or alkalinity", "Soil pH influences nutrient availability and microbial activity. The preferred range varies by crop and soil.", "Do not apply lime or sulfur based only on a generic pH cutoff; use soil-testing guidance.", "Soil Analysis"),
    ("Soil moisture", "Soil & water", "Water available in soil", "Moisture affects root activity and water stress. Measurements depend on soil texture, depth and sensor calibration.", "Too little or too much water can reduce growth; avoid treating one reading as a complete irrigation recommendation.", "Soil Analysis"),
    ("Irrigation", "Soil & water", "Applying water to crops", "Irrigation scheduling depends on weather, crop growth stage, soil water-holding capacity and effective rainfall.", "Model-based scenarios are not field-tested irrigation prescriptions.", "Yield Prediction"),
    ("NPK ratio", "Nutrients", "Relative proportions of primary macronutrients", "N, P and K represent nitrogen, phosphorus and potassium. Fertilizer labels often express N–P₂O₅–K₂O rather than elemental N–P–K.", "Check the measurement units and fertilizer label conventions before comparing quantities.", "Nutrient Advisor"),
    ("Crop yield", "Yield & analytics", "Harvested production per unit area", "Crop yield may be expressed as tonnes per hectare (t/ha), kg/ha or another unit depending on the dataset.", "Check units, crop, season and whether predictions use measured or synthetic data.", "Yield Prediction"),
    ("MAE", "Yield & analytics", "Mean absolute error", "Average absolute difference between predicted and observed values; it has the same units as yield.", "Lower is better, but compare on the same held-out dataset.", "Model Comparison"),
    ("RMSE", "Yield & analytics", "Root mean squared error", "Square root of average squared prediction error; more sensitive to large errors than MAE.", "Lower is better; compare with MAE to detect large mistakes.", "Model Comparison"),
    ("R²", "Yield & analytics", "Coefficient of determination", "Measures improvement over predicting the test-set mean under the usual R² definition. Can be negative.", "High R² on synthetic data does not prove performance on real farms.", "Model Comparison"),
    ("Overfitting", "AI & explainability", "Learning training specifics instead of general patterns", "A model can look excellent on familiar images or records yet fail on new crops, conditions or camera backgrounds.", "Use independent tests and check for duplicate or augmented-image leakage.", "Model Comparison / Smart Vision"),
    ("Transfer learning", "AI & explainability", "Reusing pretrained model features", "A pretrained network is adapted to a new task using a smaller labeled dataset.", "Its success still depends on the quality and representativeness of the target data.", "Smart Vision"),
    ("MobileNetV3", "AI & explainability", "Efficient convolutional neural network", "A compact CNN architecture used in AgriVision to classify the three trained potato-leaf categories.", "It does not recognize every crop or disease; out-of-scope images may receive confident scores.", "Smart Vision"),
    ("Grad-CAM", "AI & explainability", "Gradient-weighted Class Activation Mapping", "Visualizes regions that influence a chosen CNN class score using feature maps and gradients.", "A highlighted region is model evidence, not verified lesion segmentation or proof of disease.", "Smart Vision"),
    ("SHAP", "AI & explainability", "Shapley-based feature attribution", "Estimates how input features contribute to a model prediction relative to a reference expectation.", "Contributions describe the model's behavior; they are not causal agronomic effects.", "Yield Prediction"),
    ("Prediction interval", "AI & explainability", "Range around a numeric forecast", "A prediction interval describes uncertainty for a future outcome under specified assumptions and calibration.", "A 90% interval is a coverage target over repeated comparable cases, not a 90% guarantee for one farm.", "Yield Prediction"),
    ("Conformal prediction", "AI & explainability", "Calibration-based uncertainty method", "Uses held-out calibration errors to construct prediction sets or intervals with statistical coverage under exchangeability assumptions.", "Coverage can fail after dataset shift or if calibration and test samples are not comparable.", "Yield Prediction"),
    ("Class score", "AI & explainability", "Output of a disease classifier", "A softmax score indicates a model's relative preference among its trained classes.", "High scores are not necessarily calibrated probabilities and cannot identify an untrained disease.", "Smart Vision"),
]

DISEASES = {
    "Potato — Early Blight": {
        "agent": "Often associated with Alternaria solani / Alternaria species",
        "symptoms": "Older leaves may develop brown spots with concentric rings; yellowing can develop around lesions.",
        "favorable": "Commonly develops in warm conditions with periods of leaf wetness; stressed foliage can be more vulnerable.",
        "action": "Inspect several leaves across the field, photograph both leaf surfaces, check local extension guidance and confirm the diagnosis before treatment.",
        "prevention": "Use healthy planting material, manage crop residue, maintain balanced nutrition and minimize avoidable leaf wetness where practical.",
    },
    "Potato — Late Blight": {
        "agent": "Phytophthora infestans (oomycete)",
        "symptoms": "Water-soaked or dark lesions may spread quickly. Whitish sporulation can appear on leaf undersides in humid conditions.",
        "favorable": "Cool to moderate temperatures and high humidity / prolonged leaf wetness can promote outbreaks.",
        "action": "Treat suspected late blight as time-sensitive: isolate observations, inspect nearby plants, and promptly contact a local agricultural extension service for confirmation and appropriate management.",
        "prevention": "Monitor weather risk, use disease-free seed, inspect fields regularly and follow locally approved integrated disease-management practices.",
    },
    "Potato — Healthy": {
        "agent": "No specific disease identified among the three trained categories",
        "symptoms": "An apparently healthy leaf may be uniformly green without prominent disease lesions, but visual appearance alone does not establish overall plant health.",
        "favorable": "Not applicable.",
        "action": "Continue monitoring; check other plants, leaves and environmental conditions if symptoms are present elsewhere.",
        "prevention": "Maintain appropriate irrigation, nutrition, scouting and local disease-prevention practices.",
    },
}

FAQ = [
    ("Why are my leaves turning yellow?", "Possible causes include nutrient imbalance, excess or insufficient water, root problems, infection and natural aging. Compare older and younger leaves, inspect roots and use soil testing before acting."),
    ("Does a 99% CNN score mean the diagnosis is certainly correct?", "No. The CNN score ranks the three trained potato classes and may be overconfident on unfamiliar images. Confirm symptoms and consider expert advice."),
    ("Can the model diagnose tomato or rice?", "The current image CNN was trained for potato Early Blight, Late Blight and Healthy only. It should not be used as a general crop disease detector."),
    ("Why is my yield forecast shown as a range?", "Agricultural yield is variable. A calibrated interval expresses uncertainty alongside the point prediction, subject to its statistical assumptions."),
    ("Can I increase real yield by changing a scenario slider?", "Not necessarily. Scenario tools show how the fitted model responds to changed inputs; this is not evidence of a causal effect in the field."),
    ("How should I decide which fertilizer to apply?", "Start with crop type, stage, soil testing and local extension guidance. A leaf photograph or generic NPK rule alone cannot establish rates."),
    ("When should I harvest?", "Harvest timing depends on crop, variety, intended use, maturity and conditions. NDVI changes alone do not determine the right harvest date."),
]

TOPICS = ["All topics", "Plant health", "Nutrients", "Soil & water", "Yield & analytics", "AI & explainability"]


def _display_term(entry: tuple[str, ...]) -> None:
    name, category, subtitle, definition, caution, related = entry
    with st.expander(f"{name}  ·  {category}"):
        st.markdown(f"**{subtitle}**")
        st.write(definition)
        st.info(f"**Interpretation / limitation:** {caution}")
        st.caption(f"Related AgriVision feature: {related}")


def show() -> None:
    st.title("📚 AgriVision Knowledge Hub")
    st.markdown("Understand crop symptoms, nutrients, soil measurements and the AI behind your predictions. This is a reference guide, **not an automated field diagnosis**.")

    a, b, c = st.columns(3)
    a.metric("Knowledge topics", len(TOPICS) - 1)
    b.metric("Explained terms", len(TERMS))
    c.metric("Potato categories", len(DISEASES))

    tab_glossary, tab_disease, tab_ai, tab_faq = st.tabs([
        "📖 Search & glossary", "🥔 Potato disease guide", "🧠 Understand the AI", "❓ FAQs & sources"
    ])

    with tab_glossary:
        st.subheader("Explore agricultural concepts")
        query = st.text_input("Search the knowledge base", placeholder="Try: nitrogen, Grad-CAM, yield, chlorosis…", key="kh_search")
        topic = st.selectbox("Topic", TOPICS, key="kh_topic")
        terms = [t for t in TERMS if (topic == "All topics" or t[1] == topic) and
                 (not query.strip() or re.search(re.escape(query.strip()), " ".join(t), flags=re.IGNORECASE))]
        st.caption(f"Showing {len(terms)} of {len(TERMS)} entries")
        if terms:
            for t in terms:
                _display_term(t)
        else:
            st.warning("No matching entries. Try a broader keyword or select All topics.")

    with tab_disease:
        st.subheader("Potato leaf condition reference")
        st.caption("The current MobileNetV3 image classifier recognizes only these three categories. Other conditions are outside its training scope.")
        selected = st.selectbox("Select a condition", list(DISEASES), key="kh_disease")
        d = DISEASES[selected]
        st.markdown(f"### {selected}")
        st.markdown(f"**Typical pathogen / meaning:** {d['agent']}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Common visual signs**")
            st.write(d["symptoms"])
            st.markdown("**Conditions associated with disease**")
            st.write(d["favorable"])
        with right:
            st.markdown("**Practical next steps**")
            st.write(d["action"])
            st.markdown("**Prevention / monitoring**")
            st.write(d["prevention"])
        st.warning("The CNN output is a screening aid, not a laboratory-confirmed diagnosis. Do not select pesticides or application rates solely from an image prediction.")

    with tab_ai:
        st.subheader("How AgriVision makes predictions")
        method = st.radio("Choose an AI component", ["Smart Vision CNN", "Grad-CAM explanation", "Yield ML and SHAP", "Prediction uncertainty"], horizontal=True, key="kh_ai_method")
        if method == "Smart Vision CNN":
            st.markdown("**Workflow:** leaf photograph → resize/normalize → MobileNetV3 feature extraction → scores across three potato categories → displayed prediction.")
            st.write("The classifier learns image patterns from labeled potato leaves. Its score does not establish whether the photo is a potato leaf or whether an unseen disease is present.")
        elif method == "Grad-CAM explanation":
            st.markdown("**Workflow:** selected class score → gradient computation → weighted convolutional feature maps → spatial activation overlay.")
            st.write("Warm areas are regions influencing a particular class score. They do **not** represent measured tissue damage or a verified disease mask.")
        elif method == "Yield ML and SHAP":
            st.markdown("**Workflow:** input validation → trained regression pipeline → yield estimate → SHAP feature contributions relative to a reference expectation.")
            st.write("SHAP helps explain model behavior, not what would happen if a farmer intervened. The current AgriVision yield dataset is synthetic; field performance is not established.")
        else:
            st.markdown("**Workflow:** train/validation/calibration/test partitioning → residual-based conformal threshold → prediction interval.")
            st.write("Nominal 90% coverage refers to repeated comparable observations under conformal assumptions, not certainty about an individual field.")
        st.markdown("**Important distinction:** model performance on held-out data is different from real-world generalization. Validate against independent field observations before deployment.")

    with tab_faq:
        st.subheader("Frequently asked questions")
        for q, answer in FAQ:
            with st.expander(q):
                st.write(answer)
        st.divider()
        st.subheader("Further reading")
        st.markdown(
            "For location-specific diagnosis and crop management, consult local agricultural extension specialists. "
            "General references: [FAO](https://www.fao.org/home/en), "
            "[USDA Natural Resources Conservation Service](https://www.nrcs.usda.gov/), "
            "[CABI PlantwisePlus](https://www.cabi.org/plantwiseplus/), and "
            "[PyTorch Grad-CAM tutorial resources](https://pytorch.org/vision/stable/index.html)."
        )
        st.caption("Knowledge content is educational and not a substitute for soil testing, laboratory diagnosis or approved local agricultural advice.")
