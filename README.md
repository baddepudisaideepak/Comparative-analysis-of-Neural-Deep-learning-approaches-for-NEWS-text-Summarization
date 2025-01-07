# Comparative-analysis-of-Neural-Deep-learning-approaches-for-NEWS-text-Summarization
This project investigates and implements various neural deep learning techniques for text summarization, focusing on summarizing news articles. The project utilizes machine learning models such as Encoder-Decoder LSTM (with and without attention mechanisms) to generate concise summaries from longer news texts.

## Dataset
The dataset used is sourced from the [news_summary.csv](https://raw.githubusercontent.com/baddepudisaideepak/Comparative-analysis-of-Neural-Deep-learning-approaches-for-NEWS-text-Summarization/refs/heads/main/news_summary.csv). It includes columns for `ctext` (complete text) and `headlines` (summaries). Data preprocessing involves cleaning, tokenization, and word count analysis.

## Key Features
1. **Data Preprocessing**:
   - Cleaning and tokenizing text data.
   - Removing unnecessary characters and spaces.
   - Setting maximum sequence lengths to filter overly long sequences.
   - Visualizing word count distributions for `ctext` and `headlines`.

2. **Model Implementation**:
   - **Encoder-Decoder LSTM**:
     - Implements a sequence-to-sequence model for text summarization.
     - Includes embedding layers and LSTM layers for encoding and decoding sequences.
   - **Attention Mechanism**:
     - Enhances the LSTM model by adding an attention layer.
     - Provides context-awareness during decoding for better summarization performance.

3. **Evaluation Metrics**:
   - Uses BLEU scores and loss/accuracy metrics to evaluate model performance.
   - Visualizes training and validation loss and accuracy over epochs.

## Visualization
The project includes:
- Histograms and scatter plots of word counts for input and output text.
- WordCloud visualizations for commonly occurring terms in the dataset.

## Requirements
- Python 3.x
- TensorFlow
- NLTK
- NumPy, Pandas, Matplotlib, Seaborn
- WordCloud

## Running the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the Jupyter Notebook to preprocess the dataset, train models, and evaluate performance.

## Results
The trained models demonstrate the effectiveness of incorporating attention mechanisms for generating high-quality summaries.

## Future Work
- Extend to other datasets and domains.
- Experiment with transformer-based architectures like BERT or GPT for improved performance.

---

Let me know if you'd like any modifications or additions!
