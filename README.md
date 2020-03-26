# Lyrics-Generation
## Models for Lyrics Generation

## Model 1 - Language Model with melody embedded
In our first model we created a language model having two inputs - lyrics and the melody vector. The lyrics represent as sequences and the melody is represented by a vector with 5 different aspect of summarized melody data.

### Approach for integrating the melody information together with the lyrics
The main idea of this model is to find language model that predict the next word of a given sequence but get a shift of the prediction by adding vector of melody that need to change the prediction depended on the melody.

### Model Inputs
#### Lyrics Sequence input
Our approach is to split up the source text line-by-line, then break each line down into a series of words that build up. By that, each word will first be represented separately, and also as a part of a sequence.
The general Sequence format will be as the following:
[word1, word2, word3, ... , word10]

#### Melody input
Represent a melody, loaded with pretty_midi object and provide aspects of 5 different melody data.
- Semitone –The relative amount of each semitone across the entire song  - 12 dim
- Piano Roll - The relative amount of each piano roll across the entire song – 128 dim
- Transition Matrix  - Total pitch class transition matrix of all instruments normalize and flattened – 144
- Histogram -  The frequency of pitch classes of all instruments – dim 12
- BPM – normalize beat per minute – 1 dim
Total of 297 dimensions.


## Model 2 - Neural Machine Translation from Melody to Lyrics

