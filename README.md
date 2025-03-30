# Test Time Augmentation for Automatic Piano Transcription (TTAAPT)

## Introduction
Test Time Augmentation (TTA) is a technique used to enhance model predictions by applying transformations to input data during inference. This project explores the application of TTA to Automatic Piano Transcription (APT) by leveraging pitch shifting and time stretching techniques.

APT converts raw piano recordings into symbolic representations such as MIDI. While state-of-the-art deep learning models achieve high accuracy, they may be biased towards the datasets they were trained on. This work examines whether applying TTA with pitch and time transformations can improve transcription accuracy.

## Methodology
TTAAPT applies the following transformations to piano recordings before transcription:
- **Pitch Shifting**: Adjusting pitch by ±1 to ±3 semitones.
- **Time Stretching**: Modifying playback speed by factors between 0.9 and 1.1.

Each transformed audio file is independently transcribed. The resulting predictions are realigned and aggregated using a mode-based fusion method to enhance robustness.

## Experiments
- **Baseline Reproduction**: The frame-wise performance of a state-of-the-art APT model was reproduced using the Maestro dataset.
- **Equivariance Testing**: The APT model's sensitivity to pitch shifting and time stretching was analyzed.
- **TTA Performance Evaluation**: The impact of different augmentation intensities and ensemble sizes on transcription accuracy was measured.

### Key Findings
- Pitch shifting and time stretching introduce minor but significant performance degradation.
- TTA generally does not improve transcription accuracy and may reduce recall.
- The model is biased against augmented audio inputs, limiting TTA's effectiveness.

## Future Work
- Explore alternative augmentation techniques such as different time-stretching algorithms.
- Test more robust transcription models to mitigate augmentation bias.
- Apply TTA to out-of-domain datasets to evaluate generalization improvements.

## Dependencies
- Python 3.8+, Librosa, NumPy, SciPy
- piano_transcription_inference, MidiToolkit, Scikit-learn, Pretty MIDI  
- FFmpeg, Rubberband

## Installation
1. Install system dependencies:
   ```bash
   apt-get install -q -y libsndfile-dev rubberband-cli ffmpeg
   ```
2. Install Python dependencies:
   ```bash
   pip install librosa piano_transcription_inference miditoolkit scikit-learn pretty_midi
   ```

## Citation
If you use this work, please cite:
```
Filip Danielsson. "Test Time Augmentation for Automatic Piano Transcription." KTH Royal Institute of Technology, 2025.
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.
