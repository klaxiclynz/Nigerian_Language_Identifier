# Nigerian Language Identifier

**Author:** Bellock-Zuokemefa Emmanuel

## Problem
Many apps serving Nigerian users need to detect what language a piece of text is written in — Yoruba, Hausa, Igbo, or English — before they can translate, route, or respond to it appropriately.

## Solution
This project trains a text classifier that takes a short phrase and predicts which of the four languages it belongs to.

## Dataset
- Source: [`benjaminogbonna/nigerian_common_voice_dataset`](https://huggingface.co/datasets/benjaminogbonna/nigerian_common_voice_dataset) (Hugging Face)
- Languages: Hausa, Igbo, Yoruba, English
- Total examples: 22,295 short text phrases
- Class distribution: Hausa (9,008), Igbo (5,714), Yoruba (4,171), English (3,402)

## Approach
1. Loaded and combined text data for all four languages
2. Split data into training (80%) and testing (20%) sets
3. Converted text into numeric features using **TF-IDF**, combining:
   - Character-level patterns (1-4 character sequences) — captures spelling and diacritic differences
   - Word-level patterns (1-2 word sequences) — captures whole-word cues
4. Trained a **Logistic Regression** classifier on the combined features
5. Evaluated on the held-out test set

## Results
- **Accuracy: 99.93%** on the test set
- Precision, recall, and F1-score all ≥ 0.99 across all four languages

## Known Limitation
The model performs excellently on full, formal-length phrases (matching its training data) but shows reduced accuracy on very short, informal expressions or greetings that lack language-distinctive diacritics (e.g. "Bawo ni?", "Nagode"). This is a known challenge in short-text language identification generally, not unique to this model.

## How to Run
1. Open `Nigerian_Language_Identifier.ipynb` in Google Colab
2. Run all cells in order (Runtime → Run all)
3. Use the interactive text box near the bottom of the notebook to test your own phrases

## Files
- `Nigerian_Language_Identifier.ipynb` — full notebook (data loading, training, evaluation, demo)
- `language_model.pkl` — trained Logistic Regression model
- `char_vectorizer.pkl` — character-level TF-IDF vectorizer
- `word_vectorizer.pkl` — word-level TF-IDF vectorizer

## Tools Used
Python, pandas, scikit-learn, Google Colab
