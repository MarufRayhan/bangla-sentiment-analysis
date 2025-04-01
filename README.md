# Bangla Emotion Detection Using Deep Learning

## Overview
This repository contains the implementation of deep learning models for detecting emotions from Bangla text. The project utilizes Bidirectional Gated Recurrent Unit (BiGRU) and CNN-BiLSTM (Convolutional Neural Network with Bidirectional Long Short-Term Memory) models to classify text into six emotional categories: happiness (আনন্দ), sadness (বিষণ্ণতা), fear (ভয়), anger (রাগ), love (ভালবাসা), and surprise (আশ্চর্য). The research paper associated with this work was published in the 2020 23rd International Conference on Computer and Information Technology (ICCIT).

## Features
- Text preprocessing pipeline optimized for Bangla language
- Implementation of two deep learning architectures:
  - CNN-BiLSTM: Combines Convolutional Neural Networks with Bidirectional Long Short-Term Memory
  - BiGRU: Utilizes Bidirectional Gated Recurrent Units
- Multilabel emotion classification with six emotional categories
- Comprehensive evaluation using accuracy, precision, recall, and F1-score
- Support for Google-translated Bangla text, making it useful for non-native speakers

## Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

## Running the Notebook
This project is provided as a Jupyter notebook, which can be run in:
- Google Colab
- Jupyter Notebook
- JupyterLab

No installation is required - simply upload the notebook to your preferred environment and run the cells sequentially.

## Dataset
The models are trained on a Bangla text dataset encompassing six emotion categories. The dataset was created by translating an English emotion dataset using Google Translator, comprising 7,214 sentences with the following distribution:

| Emotion | Bangla | Number of Sentences |
|---------|--------|---------------------|
| Happy | আনন্দ | 1,362 |
| Sad | বিষণ্ণতা | 1,352 |
| Fear | ভয় | 1,395 |
| Angry | রাগ | 1,373 |
| Love | ভালবাসা | 1,199 |
| Surprise | আশ্চর্য | 533 |

## Model Architectures

### CNN-BiLSTM
This model combines Convolutional Neural Networks with Bidirectional Long Short-Term Memory networks:
- Embedding layer with 64 dimensions
- Two 1D convolutional layers with dropout
- Max pooling layer
- BiLSTM layer with dropout
- Fully connected layer and softmax activation for classification

### BiGRU
This model utilizes Bidirectional Gated Recurrent Units:
- Embedding layer with 64 dimensions
- BiGRU cell units with dropout and RELU activation
- Fully connected layer and softmax activation for classification

## Performance
The models were evaluated based on accuracy, precision, recall, and F1-score:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CNN-BiLSTM | 66.62% | 68.20% | 66.63% | 67.41% |
| BiGRU | 64.96% | 65.30% | 65.55% | 65.42% |

Detailed per-emotion performance for CNN-BiLSTM:

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Happy (আনন্দ) | 52.35% | 71.20% | 60.34% |
| Sad (বিষণ্ণতা) | 62.60% | 56.62% | 59.46% |
| Fear (ভয়) | 72.50% | 72.59% | 72.50% |
| Angry (রাগ) | 71.43% | 63.43% | 67.19% |
| Love (ভালবাসা) | 77.78% | 68.75% | 72.99% |
| Surprise (আশ্চর্য) | 72.55% | 67.27% | 69.81% |

## Significance
This research is particularly valuable for:
- Understanding emotions from Bangla text for native speakers
- Helping non-native Bangla users interpret emotions in translated text
- Advancing NLP research in the Bangla language, which is spoken by approximately 210 million people worldwide

## Future Work
- Building a more comprehensive Bangla emotion corpus with proper annotations
- Implementing additional deep learning approaches to extract more semantic and syntactic features
- Improving classification accuracy through model optimization
- Supporting Bangla dialects and variations

## Citation
If you use this work in your research, please cite:
```
@inproceedings{rayhan2020multilabel,
  title={Multilabel Emotion Detection from Bangla Text Using BiGRU and CNN-BiLSTM},
  author={Rayhan, Md Maruf and Musabe, Taif Al and Islam, Md Arafatul},
  booktitle={2020 23rd International Conference on Computer and Information Technology (ICCIT)},
  pages={1--6},
  year={2020},
  organization={IEEE},
  doi={10.1109/ICCIT51783.2020.9392690}
}
```

## License
This project is licensed under the MIT License.

## Contact
For any queries, please open an issue in this repository or contact me at maruf.rayhan14@gmail.com.
