"""Streamlit image diagnosis page using trained MobileNetV3, not OpenCV rules."""
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from modules.image_detection import analyse_crop_image, DEFAULT_MODEL


def show():
    st.title('📷 ML-based Leaf Disease Detection')
    st.caption('MobileNetV3 image classifier. Predictions are advisory, not a definitive plant diagnosis.')
    uploaded=st.file_uploader('Upload leaf photograph',type=['jpg','jpeg','png','webp'])
    if uploaded is None:
        st.info('Upload a clear leaf photograph to begin. The trained model must be installed first.')
        return
    if not Path(DEFAULT_MODEL).exists():
        st.error('Trained model not found. Train it using train_image_model.py and place the resulting .pt file in artifacts/.')
        return
    try:
        image=Image.open(uploaded).convert('RGB')
        st.image(image,caption='Uploaded leaf',use_container_width=True)
        result=analyse_crop_image(image)
    except (UnidentifiedImageError,ValueError,RuntimeError,FileNotFoundError) as exc:
        st.error(f'Unable to classify image: {exc}');return
    st.subheader('Prediction')
    st.metric('Top predicted class',result['disease'])
    st.metric('Model softmax score',f"{result['confidence']:.1f}%")
    if result['needs_review']:
        st.warning('Low model score: seek additional review. The 70% threshold is provisional, not calibrated.')
    st.caption(result['note'])
    table=pd.DataFrame(result['predictions'])
    st.bar_chart(table.set_index('disease')['probability'])
    st.download_button('Download prediction CSV',table.to_csv(index=False),file_name='leaf_predictions.csv',mime='text/csv')
    st.info('For treatment decisions, verify crop species and symptoms with a qualified agronomist. This model does not diagnose unseen classes.')