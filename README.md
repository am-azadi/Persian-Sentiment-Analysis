# Persian Text-Based Emotion Detection

## Overview
This project focuses on the challenging task of emotion detection in Persian text, aiming to enhance natural language understanding capabilities for applications in sentiment analysis, social media monitoring, and user engagement analysis in Persian-speaking communities. Emotion detection plays a crucial role in understanding the affective aspects of text data, providing valuable insights into users' sentiments and emotional states.

## Authors
- AmirMohammd Azadi
- Sina Zamani

## Abstract
Emotion detection has become a critical component in natural language processing, offering deeper insights into the emotional context of textual data. This project addresses the gap in the literature for languages with non-Latin scripts, such as Persian, by proposing and implementing an emotion detection model specifically tailored for Persian text.

## Datasets
We have used ArmanEmo, a human-labeled emotion dataset of more than 7000 Persian sentences labeled for seven categories. The dataset has been collected from different resources, including Twitter, Instagram, and Digikala comments. Labels are based on Ekman’s six basic emotions (Anger, Fear, Happiness, Hatred, Sadness, Wonder) and another category (Other).

## Preprocessing
1. Assign 'text' as the title of the column of sentences and 'label' as the title of the column of labels.
2. Map the labels to numbers from 0 to 6.
3. Prepare the data for training by tokenizing, truncating, and padding, while limiting the maximum length for sentences to 128.

## Modeling
We use the following models:
1. **ParsBERT**: A monolingual language model based on BERT architecture, fine-tuned on Persian datasets.
2. **XLM-RoBERTa**: A multilingual transformer-based model pre-trained on text in 100 languages, evaluated in two variations (XLM-RoBERTa-base and XLM-RoBERTa-large).

### Training
- **Zero-shot learning**: Enables models to generalize to new and unseen classes without explicit training.
- **Fine-tuning**: Adapting the pre-trained models to the specific linguistic nuances and emotional expressions present in Persian.

### Hyperparameters
- Learning rate: 1e-5
- Epochs: 3

## Results
The fine-tuned XLM-RoBERTa-large model significantly outperforms other models on the test set in terms of average macro F1 score, precision, recall, and accuracy.

| Model                     | Precision (Macro) | Recall (Macro) | F1 (Macro) | Accuracy |
|---------------------------|-------------------|----------------|------------|----------|
| ParsBERT (zero-shot)      | 0.13              | 0.15           | 0.08       | 0.15     |
| XLM-RoBERTa-base (zero-shot)  | 0.03              | 0.14           | 0.05       | 0.23     |
| XLM-RoBERTa-large (zero-shot) | 0.02              | 0.14           | 0.03       | 0.13     |
| ParsBERT (trained)        | 0.68              | 0.64           | 0.64       | 0.65     |
| XLM-RoBERTa-base (trained)    | 0.65              | 0.66           | 0.65       | 0.66     |
| XLM-RoBERTa-large (trained)   | 0.76              | 0.73           | 0.74       | 0.75     |

## Conclusion
The XLM-RoBERTa-large model achieves the highest performance in emotion detection from Persian text. Further research is needed to address challenges with sentences carrying mixed emotions.

## References
1. Hossein Mirzaee, Javad Peymanfard, Hamid Habibzadeh Moshtaghin, Hossein Zeinali, ArmanEmo: A Persian Dataset for Text-based Emotion Detection.
2. Mehrdad Farahani, Mohammad Gharachorloo, Marzieh Farahani, and Mohammad Manthouri. Parsbert: Transformer-based model for Persian language understanding. Neural Processing Letters, 53(6):3831–3847, 2021.
