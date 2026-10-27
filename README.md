# ✈️ Aircraft Damage Classification & Captioning

Automated visual inspection of aircraft surfaces with pretrained deep-learning models:

- **Classification** — VGG16 (ImageNet weights, frozen) + a custom dense head classifies surface damage as **dent** or **crack**.
- **Captioning** — Salesforce **BLIP**, wrapped in a custom Keras layer, writes natural-language captions and summaries of damage images.

## Results

| Metric | Value |
|---|---|
| Validation accuracy (5 epochs) | 76.0% |
| Test accuracy | 78.1% |
| Training data | 300 images, 2 classes |

<p align="center">
  <img src="images/accuracy_curve.png" width="420" alt="Training and validation accuracy">
  <img src="images/sample_prediction.png" width="320" alt="Sample prediction">
</p>

**BLIP output examples**

| Mode | Generated text |
|---|---|
| Caption | "this is a picture of a plane that was sitting on the ground in a field" |
| Summary | "this is a detailed photo showing the damage to the fuselage of the aircraft" |
| Summary | "this is a detailed photo showing the engine of a boeing 747" |

## How it works

```
Image (224×224) → VGG16 conv base (frozen) → Flatten
               → Dense(512, ReLU) → Dropout(0.3)
               → Dense(512, ReLU) → Dropout(0.3)
               → Dense(1, sigmoid)  →  dent / crack
```

Transfer learning makes this workable with only a few hundred images: the pretrained convolutional layers already detect edges, textures and shapes, so only the small head needs training.

For captioning, `BlipCaptionSummaryLayer` subclasses `tf.keras.layers.Layer` and calls the PyTorch BLIP model through `tf.py_function`, using a different text prompt for "caption" vs "summary" mode.

## Run it

```bash
git clone https://github.com/<rafayelmirijanyan1997>/aircraft-damage-classification.git
cd aircraft-damage-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook aircraft_damage_classification.ipynb
```

The notebook downloads the dataset automatically (~ a few MB). Training runs on CPU in a few minutes; the BLIP model (~1 GB) downloads from Hugging Face on first use.

## Project structure

```
├── aircraft_damage_classification.ipynb   # full pipeline with outputs
├── images/                                # plots used in this README
├── requirements.txt
└── README.md
```

## Next steps

- Data augmentation and early stopping
- Fine-tune the top VGG16 block or try EfficientNet / ResNet50
- Precision, recall and a confusion matrix (a missed crack costs more than a false alarm)
- Fine-tune BLIP on inspection-specific captions

## Tech

Python · TensorFlow / Keras · PyTorch · Hugging Face Transformers · Matplotlib

## Acknowledgments

- Dataset: [Aircraft Damage Detection](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk), Roboflow Universe, CC BY 4.0
- BLIP: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Started from an IBM Skills Network deep learning course project and extended here.
