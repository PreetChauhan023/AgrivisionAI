"""Detailed MobileNetV3 potato leaf diagnosis with optional true Grad-CAM."""
from io import BytesIO
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
from modules.image_detection import analyse_crop_image

DETAILS = {
    'Potato___Early_blight': {
        'title': 'Early blight', 'organism': 'Typically associated with Alternaria fungi',
        'overview': 'Early blight commonly causes brown, often target-like lesions. Older foliage may be affected first; yellowing can occur around lesions.',
        'visual': ['Brown lesions that may develop concentric rings', 'Yellowing around spots', 'Progression from older leaves in some cases'],
        'response': ['Inspect several leaves across the plant, especially older foliage.', 'Document lesion progression and remove infected plant debris when appropriate.', 'Seek local extension guidance if symptoms expand or the crop is economically important.'],
        'prevention': ['Avoid prolonged periods of wet foliage where possible.', 'Maintain field sanitation and balanced crop nutrition.', 'Follow locally recommended disease monitoring practices.'],
        'caution': 'Similar lesions can have other causes; image classification alone does not confirm the pathogen.'},
    'Potato___Late_blight': {
        'title': 'Late blight', 'organism': 'Associated with Phytophthora infestans (an oomycete)',
        'overview': 'Late blight can cause rapidly spreading dark, water-soaked lesions, particularly during moist and cool conditions.',
        'visual': ['Irregular dark or water-soaked leaf areas', 'Rapid spread under favorable weather conditions', 'Possible pale sporulation on undersides in humid conditions'],
        'response': ['Inspect neighboring plants and both sides of affected leaves promptly.', 'Avoid moving potentially infected material between plots.', 'Contact a local agricultural extension specialist promptly for confirmation and management advice.'],
        'prevention': ['Monitor periods of high humidity and prolonged leaf wetness.', 'Use locally recommended disease-alert and sanitation practices.', 'Follow locally approved management advice for the crop and region.'],
        'caution': 'Late blight can progress quickly. Do not base chemical-treatment decisions solely on an image classification.'},
    'Potato___healthy': {
        'title': 'Healthy-looking leaf', 'organism': 'No early- or late-blight pattern recognized by this classifier',
        'overview': 'The model selected the healthy class among its three possible classes. This does not exclude other pathogens, pests, deficiencies, or damage.',
        'visual': ['Absence of obvious disease patterns in this photograph', 'Relatively uniform tissue in the visible region'],
        'response': ['Check other leaves and the entire plant for symptoms outside this photograph.', 'Continue routine crop monitoring.', 'If the plant is declining despite this result, consult a local expert.'],
        'prevention': ['Maintain good field sanitation.', 'Use crop-appropriate irrigation and nutrition practices.', 'Inspect leaves regularly.'],
        'caution': 'This is not a comprehensive plant-health assessment.'},
}


def _nice(label):
    return str(label).replace('___', ' — ').replace('_', ' ')


def _score_rows(result):
    rows = []
    for p in result.get('predictions') or []:
        if not isinstance(p, dict):
            continue
        try:
            score = float(p.get('probability', 0))
        except (ValueError, TypeError):
            continue
        if 0 <= score <= 1:
            rows.append({'Condition': _nice(p.get('disease', 'Unknown')), 'Score (%)': score * 100})
    return sorted(rows, key=lambda x: -x['Score (%)'])


def _quality(image):
    rgb = image.convert('RGB')
    gray = np.asarray(rgb.resize((256,256)).convert('L'), dtype=np.float32)
    brightness = float(gray.mean())
    # Simple edge-gradient statistic for blur *screening*, not a verified blur detector.
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return {'resolution': f'{image.width} × {image.height}',
            'brightness': round(brightness, 1), 'detail_index': round(float((dx+dy)/2), 1),
            'warnings': (["Image appears quite dark; use better lighting."] if brightness < 55 else []) +
                        (["Image appears very bright; avoid glare and overexposure."] if brightness > 210 else []) +
                        (["Image is small; capture a closer, sharper leaf image."] if min(image.size) < 224 else [])}


def _reset_analysis_when_new_image(digest):
    if st.session_state.get('sv_digest') != digest:
        st.session_state['sv_digest'] = digest
        st.session_state.pop('sv_result', None)
        st.session_state.pop('sv_explanation', None)


def show():
    st.title('AI Smart Vision Detailed Leaf Analysis')
    st.write('Deep-learning classification ')

    left, right = st.columns([1.15, 1])
    with left:
        uploaded = st.file_uploader('Upload a potato leaf photo', type=['jpg','jpeg','png'], key='smart_vision_leaf')
        if uploaded is None:
            st.info('Upload a photo to begin the analysis.')
            return
        raw = uploaded.getvalue()
        try:
            image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert('RGB')
            if min(image.size) < 64:
                st.error('Image must be at least 64 × 64 pixels.')
                return
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f'Unable to read image: {exc}')
            return
        st.image(image, caption='Uploaded leaf image', use_container_width=True)
    with right:
        st.subheader('Analysis configuration')
        st.write('**Model:** MobileNetV3-Small (trained CNN)')
        st.write('**Crop:** Potato only')
        st.write('**Tasks:** Disease class ranking, image-quality screening, model explanation and guidance')

        explain_requested = st.checkbox('Generate Grad-CAM explanation', value=True,
                                         help='Visualizes regions that contributed to the CNN score; does not detect exact lesions.')
        run = st.button('Analyze potato leaf', type='primary', use_container_width=True)

    digest = hashlib.sha256(raw).hexdigest()
    _reset_analysis_when_new_image(digest)
    if run:
        with st.spinner('Running CNN inference...'):
            try:
                result = analyse_crop_image(image)
                if not isinstance(result, dict) or not result.get('disease'):
                    raise ValueError('CNN returned no valid disease class.')
                st.session_state['sv_result'] = result
                st.session_state.pop('sv_explanation', None)
            except (OSError, RuntimeError, ValueError, KeyError, FileNotFoundError) as exc:
                st.error(f'CNN prediction failed: {exc}')
                return
        if explain_requested:
            with st.spinner('Calculating Grad-CAM activations...'):
                try:
                    from modules.gradcam_explain import explain_leaf
                    st.session_state['sv_explanation'] = explain_leaf(image, target_label=result['disease'])
                except Exception as exc:
                    st.session_state['sv_explanation'] = {'error': str(exc)}

    result = st.session_state.get('sv_result')
    if result is None:
        return
    label = str(result['disease'])
    try:
        score = float(result.get('confidence', 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0.0, min(100.0, score))
    details = DETAILS.get(label)
    st.divider()
    st.subheader('1. Prediction summary')
    col_a, col_b, col_c = st.columns([2,1,1])
    col_a.metric('Top-ranked condition', _nice(label))
    col_b.metric('CNN class score', f'{score:.2f}%')
    col_c.metric('Classes evaluated', str(len(result.get('predictions') or [])))
    st.caption('This percentage is an uncalibrated softmax score, not the probability that the medical/agricultural diagnosis is correct.')
    if result.get('needs_review') or score < 70:
        st.warning('Lower model score: obtain a clearer image and seek expert review.')
    if score >= 95:
        st.info('High model score indicates strong class preference on this image; it does not guarantee correctness on real-world field images.')

    overview, visual, details_tab, methodology = st.tabs([
        'Full prediction report', ' Explainable AI', ' Disease & crop guidance', '️ Model methodology'])

    with overview:
        st.subheader('Class-by-class CNN output')
        rows = _score_rows(result)
        if rows:
            df = pd.DataFrame(rows)
            st.bar_chart(df.set_index('Condition')['Score (%)'])
            st.dataframe(df.style.format({'Score (%)':'{:.2f}'}), use_container_width=True, hide_index=True)
            if len(rows) >= 2:
                st.write(f"**Top-two separation:** {rows[0]['Score (%)'] - rows[1]['Score (%)']:.2f} percentage points")
            st.caption('Scores are normalized across only the three trained classes; unknown diseases are not an explicit category.')
        else:
            st.info('Class-score breakdown was not returned by the inference module.')
        st.subheader('Uploaded-image quality checks')
        q = _quality(image)
        q1,q2,q3 = st.columns(3)
        q1.metric('Resolution', q['resolution'])
        q2.metric('Mean brightness (0–255)', q['brightness'])
        q3.metric('Image detail index', q['detail_index'])
        st.caption('Brightness and detail index are descriptive image statistics, not disease indicators or clinically validated blur tests.')
        for warning in q['warnings']:
            st.warning(warning)
        if not q['warnings']:
            st.write('No obvious resolution or brightness warning was triggered by the basic checks.')

    with visual:
        st.subheader('What parts of the image influenced the CNN?')
        st.write('Grad-CAM uses gradients from the final convolutional block to visualize regions associated with the selected class score.')
        explanation = st.session_state.get('sv_explanation')
        if explanation and 'overlay' in explanation:
            orig, attn = st.columns(2)
            orig.image(image, caption='Original image', use_container_width=True)
            attn.image(explanation['overlay'], caption=f"Grad-CAM overlay for {_nice(explanation['label'])}", use_container_width=True)
            st.image(explanation['heatmap'], caption='Relative class activation: warmer tones indicate greater Grad-CAM activation', width=300)
            st.info('This is a **model attention visualization**, not confirmed disease segmentation. Highlighted regions do not prove lesions or identify the pathogen.')
        elif explanation and 'error' in explanation:
            st.warning(f"Grad-CAM could not be generated: {explanation['error']}")
        else:
            st.info('Enable “Generate Grad-CAM explanation” and click Analyze to view a genuine model-based heatmap.')
        st.markdown('**Interpreting the explanation:** Look for whether highlighted areas overlap the leaf rather than the background. Background attention may be a warning that the model uses spurious visual cues. A convincing-looking heatmap is not evidence of model accuracy.')

    with details_tab:
        if details:
            st.subheader(details['title'])
            st.write('**Commonly associated agent:**', details['organism'])
            st.write(details['overview'])
            st.markdown('#### Typical symptoms to verify manually')
            for text in details['visual']:
                st.markdown(f'- {text}')
            st.markdown('#### Recommended next steps')
            for text in details['response']:
                st.markdown(f'- {text}')
            st.markdown('#### Prevention and monitoring')
            for text in details['prevention']:
                st.markdown(f'- {text}')
            st.warning(details['caution'])
        else:
            st.info('No condition-specific agricultural guidance is mapped for this class.')
        st.caption('Guidance is general educational material, not individualized pesticide or fertilizer prescriptions. Seek locally qualified agricultural advice before treating crops.')

    with methodology:
        st.markdown('''**How the AI works**

1. The uploaded RGB image is resized to **224 × 224** and normalized using ImageNet channel statistics.
2. A pretrained **MobileNetV3-Small** backbone extracts visual representations.
3. The fine-tuned classification head produces logits for the three trained potato categories.
4. Softmax converts those logits to relative class scores; the highest-ranked class is displayed.
5. Optional **Grad-CAM** backpropagates the selected class logit through the last convolutional block and creates a relative activation map.

**Known limitations:** Controlled-background PlantVillage photos, only three potato conditions, no unknown-class detector, no confirmed severity grading, and no calibrated certainty. A strong test-set result does not guarantee field performance.''')
        st.write('**Inference model:**', result.get('model_type', 'MobileNetV3-Small'))
        if result.get('note'):
            st.caption(str(result['note']))
