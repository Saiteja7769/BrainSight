# BrainSight

Explainable brain MRI tumor classification research prototype using a
pretrained ResNet18 model, PyTorch, Grad-CAM, and Streamlit.

## Classes
- Glioma
- Meningioma
- Pituitary

## Structure
```text
BrainSight/
├── app/
│   ├── app.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── explainability.py
│   └── evaluation.py
├── models/
│   └── best_resnet18.pth
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

Keep the existing `models/best_resnet18.pth`. The loader expects the checkpoint
to contain `model_state_dict`, matching the training notebook.

## Windows setup
Run from the BrainSight project root:
```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Your current environment already has CUDA-enabled PyTorch and torchvision.
Do not reinstall them blindly. If needed, select a compatible build using the
official PyTorch installation selector.

## Run
```powershell
python -m streamlit run app\app.py
```

## Tests
```powershell
python -m pytest tests -q
```

## Recorded baseline results
Held-out test set: 542 images.
- Accuracy: 95.02%
- Macro precision: 93.67%
- Macro recall: 94.82%
- Macro F1: 93.98%
- Weighted F1: 94.90%

Confusion matrix (rows = actual, columns = predicted):
```text
              Glioma  Meningioma  Pituitary
Glioma           242           3          0
Meningioma         7         105          0
Pituitary          9           8        168
```

These are dataset-specific research results, not evidence of clinical validity.
BrainSight is not a medical diagnostic tool. Softmax scores are not calibrated
disease probabilities. Grad-CAM is not segmentation and does not prove tumor
localization. External validation has not been established.
