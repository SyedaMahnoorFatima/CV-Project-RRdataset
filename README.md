# Joint Detection of AI-Generated Images and Post-Processing Alterations

Computer Vision Project, Sapienza University of Rome, A.A. 2025-2026  
Course: Computer Vision, Prof. Irene Amerini  
Team: Mahnoor Fatima Syeda 2283844, Naveen Tiwari 2261826

## Project Overview

This project studies a practical image forensics problem: detecting whether an image is real or AI-generated while also identifying whether it has been altered by a common post-processing operation.

Most AI-generated image detectors are tested in clean and controlled conditions. In real-world settings, however, images are often uploaded to social media, compressed, resized, re-shared, or photographed from a screen. These changes can weaken or hide the traces that a detector normally uses. For this reason, the project does not treat AI detection and post-processing detection as two separate problems. Instead, it uses a shared model that learns both tasks at the same time.

The model receives one image and produces two predictions:

- a binary label: `real` or `AI-generated`
- a transformation label: `original`, `internet-transmitted`, or `re-digitized`

The goal is not only to obtain a working classifier, but also to understand how the two tasks influence each other and which transformations make real/fake detection more difficult.

## Dataset

The project uses RRDataset, introduced by Li et al. in their ICCV 2025 work on benchmarking AI-generated image detection in real-world challenging scenarios.

The dataset contains both real and AI-generated images under three transformation categories:

- `original`: images without the additional post-processing considered in this project
- `transfer`: images affected by internet transmission or platform-style processing
- `redigital`: images that have gone through re-digitization, such as screen recapture

### Dataset Access

The full RRDataset used in this project is large, so it is not included directly in this GitHub repository.  
It can be accessed from the following Google Drive link:

[Download RRDataset from Google Drive](https://drive.google.com/drive/folders/1zn6zR5m97mCM5KgNVFEP3e1hBkXLjHRo)

After downloading, place the dataset in the following structure before running the notebook:

```text
RRDataset_final/
  original/
    real/
    ai/
  transfer/
    real/
    ai/
  redigital/
    real/
    ai/
```

During preprocessing, the notebook builds a single inventory of all images and creates stratified train, validation, and test splits. The split is based on the combined real/fake and transformation labels, so each subset keeps a balanced representation of the six cross-classes.

## Method

The project follows a multi-task learning setup. A shared visual backbone extracts image features, and two independent classification heads use those features for the two tasks:

- `fc_rf`: real/fake classification head
- `fc_tr`: transformation classification head

Two backbone families are used:

- ResNet50 as a simple convolutional baseline
- Vision Transformer as the main model family

The notebook includes support for the Google ViT model used in the course material. For local testing and faster experiments, it also uses `vit_small_patch16_224` from `timm`, which keeps the same transformer-based direction while being lighter to run.

The main multi-task experiments compare different loss-weighting strategies:

- real/fake only
- transformation only
- equal task weighting
- real/fake-heavy weighting
- transformation-heavy weighting
- learned uncertainty weighting based on Kendall et al.

This allows the project to test whether the two objectives support each other or compete for representation capacity.

## Notebook Structure

The notebook is organized to follow the required project structure:

- imports and package setup
- global configuration and dataset paths
- helper functions
- dataset inventory and train/validation/test split generation
- dataset class and image transformations
- model definition
- training loops
- baseline experiments
- multi-task ablation experiments
- final evaluation
- confusion matrices, Pareto-style comparison, Grad-CAM visualization, and cross-class analysis

Generated files are written under:

```text
splits/
results/
results/checkpoints/
```

These folders are created automatically by the notebook when the corresponding cells are run.

## Requirements

The project is implemented in PyTorch. The main Python packages are:

```text
torch
torchvision
timm
transformers
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

When the notebook is run in Google Colab, the setup cell installs the extra packages needed for the experiment.

## How to Run

1. Open the notebook in Google Colab, Kaggle, or a local Jupyter environment.
2. Download the dataset from the Google Drive link in the Dataset section and place it in the expected folder structure.
3. Update `DATASET_ROOT` in the paths cell if the dataset is stored somewhere else.
4. Run the cells in order.
5. For a quick local check, keep `LOCAL_DEBUG = True`.
6. For the final experiment, set `LOCAL_DEBUG = False` and use a GPU runtime.

The notebook contains conservative defaults for local testing because the full dataset and transformer models can be expensive to train without GPU acceleration.

## Current Experimental Results

The executed notebook currently reports results from a debug-sized run. This was useful for verifying the full pipeline, checking that the data loading works correctly, and comparing the different training configurations before running longer experiments.

Baseline validation results:

| Model | Task setup | Real/Fake Acc. | Real/Fake F1 | Transform Acc. | Transform F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet50 | real/fake only | 0.6200 | 0.5786 | - | - |
| ResNet50 | transformation setup | 0.5200 | 0.5113 | 0.4067 | 0.3883 |
| ViT small | real/fake only | 0.8200 | 0.8193 | - | - |
| ViT small | transformation setup | 0.5067 | 0.4504 | 0.4800 | 0.4676 |

Multi-task ablation results:

| Run | Loss weighting | Real/Fake Acc. | Real/Fake F1 | Transform Acc. | Transform F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| real/fake only | 1.0 / 0.0 | 0.8200 | 0.8193 | 0.3400 | 0.2820 |
| transformation only | 0.0 / 1.0 | 0.5067 | 0.4504 | 0.4800 | 0.4676 |
| equal weighting | 0.5 / 0.5 | 0.8200 | 0.8199 | 0.5000 | 0.4994 |
| real/fake-heavy | 0.7 / 0.3 | 0.7800 | 0.7792 | 0.4400 | 0.4297 |
| transformation-heavy | 0.3 / 0.7 | 0.7600 | 0.7596 | 0.4933 | 0.4682 |
| uncertainty weighting | learned / learned | 0.8067 | 0.8065 | 0.5067 | 0.5081 |

Final evaluation from the selected model in the executed notebook:

- real/fake accuracy: `0.8367`
- real/fake macro F1: `0.8357`
- transformation accuracy: `0.4833`
- transformation macro F1: `0.4789`

The cross-class breakdown showed that AI-generated images were generally easier to classify than real images across the tested transformation groups. The gap was largest for the original category and smaller for re-digitized images:

| Transformation | AI accuracy | Real accuracy |
| --- | ---: | ---: |
| original | 0.9767 | 0.6154 |
| transfer | 0.9778 | 0.7414 |
| redigital | 0.9273 | 0.8298 |

These results should be interpreted as pipeline and experiment results from the current debug run, not as final full-scale benchmark numbers.

## Main Observations

The Vision Transformer baseline performed better than the ResNet50 baseline on the real/fake task in the current run. This supports the idea that global patch-level attention can be useful for forensic cues that are not always localized in one small region of the image.

The transformation task was harder than binary real/fake detection. This is expected because post-processing categories can share visual effects, and some transformations may damage or hide the traces used for the real/fake decision.

The loss-weighting experiments show a clear trade-off. Giving all weight to one task weakens the other head, while balanced or learned weighting gives a more useful compromise. In the current run, uncertainty weighting produced the best transformation F1 among the multi-task settings while keeping real/fake accuracy high.

## Limitations

The current executed results were produced with debug settings. A full final run should use the complete dataset, a GPU runtime, and a larger number of epochs.

The transformation classifier still has room for improvement. More training time, stronger data augmentation, and a full Google ViT run may improve this part of the model.

The project focuses on the provided RRDataset categories. Performance on other social media pipelines, compression settings, camera recapture conditions, or generative models would need additional testing.

## Future Work

Useful next steps would include:

- running the complete training configuration on the full dataset
- comparing the lightweight ViT with the full Google ViT backbone
- adding frequency-domain features or compression-aware augmentations
- testing whether separate task-specific layers improve the transformation head
- evaluating the model on external images from unseen sources

## References

Li, Chunxiao, et al. "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios." Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

Kendall, Alex, Yarin Gal, and Roberto Cipolla. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.

