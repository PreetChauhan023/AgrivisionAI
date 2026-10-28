# AgriVision AI — ML image disease detection

Replace existing `modules/image_detection.py` and `modules/page_image_detection.py` with the supplied files. Put `train_image_model.py` in the project root, alongside `app.py`. Existing app navigation can remain unchanged if it already calls `modules.page_image_detection.show()`.

Install additional dependencies:

```bash
pip install torch torchvision
```

Download the actual **PlantVillage leaf images** (not merely the CSV index). Create **disjoint** train/val/test splits (suggested 70/15/15) using the same class folder names:

```
data/leaf_images/
  train/Apple___Apple_scab/*.jpg
  train/Apple___healthy/*.jpg
  val/Apple___Apple_scab/*.jpg
  val/Apple___healthy/*.jpg
  test/Apple___Apple_scab/*.jpg
  test/Apple___healthy/*.jpg
```

Repeat for all selected classes. Ideally split by plant or acquisition session to prevent near-duplicate leakage. Keep test split fixed and untouched during training. Each split must include folders for all classes. Do not train on `mapped_dataset1.csv`: its environmental rows are not genuinely paired with the images.

```bash
python train_image_model.py --data data/leaf_images --epochs 10
streamlit run app.py
```

Model output: `artifacts/leaf_disease_mobilenet.pt` and `artifacts/leaf_disease_mobilenet.metrics.json`. Initial training may download pretrained ImageNet weights; use `--scratch` to disable that. Training may be slow without GPU.

**Limitations:** No trained weights are included. No accuracy is claimed before training and testing. A softmax score is not calibrated confidence. The 70% review rule is a provisional UX safeguard, not statistically validated selective prediction. PlantVillage contains controlled images, so evaluate on separate real-field data (e.g. PlantDoc with matched classes) before claiming real-world robustness. Old heuristic treatment suggestions are deliberately not automatically attached to CNN labels because disease names and crops require species-specific guidance.
