# LOREM: Language-consistent Open Relation Extraction from Unstructured Text

For a detailed description of the model, we refer you to our paper “LOREM: Language-consistent Open Relation Extraction from Unstructured Text”. Please cite this work if it was useful to you.
> Tom Harting, Sepideh Mesbah, and Christoph Lofi. 2020. LOREM: Language-consistent Open Relation Extraction from Unstructured Text. In _Proceedings of The Web Conference 2020 (WWW ’20)_. Association for Computing Machinery, New York, NY, USA, 1830–1838. DOI:https://doi.org/10.1145/3366423.3380252

## Table of contents
- [Description](#description)
- [Quick start](#quick-start)
    * [Training and testing](#training-and-testing)
        + [Language-individual model](#language-individual-model)
        + [Language-consistent model](#language-consistent-model)
        + [LOREM](#lorem)
    * [Extracting relations](#extracting-relations)
- [Data format](#data-format)
    * [Word embeddings](#word-embeddings)
    * [Training and test sentences](#training-and-test-sentences)
    * [Exported predictions](#exported-predictions)
- [References](#references)

## Description
LOREM is the first language-consistent open relation extraction model. It can be used to extract relations from multilingual text corpora.
For example, given the sentence
> *"During world war II, Alan Turing deciphered the Enigma machine which was used to send secret messages.”*

and the entity tuple
> <Alan Turing, Enigma machine>

LOREM is able to correctly extract the relation as being
> deciphered

## Quick start
In order to use LOREM, you first need to install the required packages from _requirements.txt_. Once this is done, we start by training and testing the model.

### Training and testing
As is described in our paper, LOREM includes two sub-models; language-consistent and language-individual models.
We need to train and test both models separately before we can combine them to form the full LOREM model.

#### Language-individual model
1. Download pre-trained word embeddings or train your own for the current language (f.e. from https://fasttext.cc). The required format is described in section [Data format](#data-format).
2. Make sure that the training sentences are formatted correctly. The required format is described in section [Data format](#data-format).
3. Open the file _LanguageIndividualModel.py_ and define the following variables:
    * The model type (use a CNN, LSTM or both).
    * The current language abbreviation.
    * The number of word embeddings that should be used.
    * The file paths to;
        * the word embedding file,
        * the file where the created data file should be stored,
        * the file where the trained model weights will be stored,
        * an optional model from which transfer learning should be employed (this is still under construction),
        * the training sentences and their entity files and
        * the test sentences and their entity files.
    * The test, transfer-learning and retrain options.
4. Upon running the file, a new data file will be created in which all the needed data is properly stored (if it does not yet exist). Then, a new language-individual model will be trained (if it does not yet exist). The weights of the trained model are stored in the defined .h5 file.
5. If the test option is set to True, the newly trained model will be tested on the defined test sentences. The system outputs a precision, recall and F1-score.
6. Repeat this step for every language.

#### Language-consistent model
1. Download pre-trained (multilingual) word embeddings or train your own for the current languages (f.e. from https://fasttext.cc). The required format is described in section [Data format](#data-format).
2. Make sure that the training sentences for all separate languages are formatted correctly. The required format is described in section [Data format](#data-format).
3. Open the file _LanguageConsistentModel.py_ and define the following variables:
    * The model type (use a CNN, LSTM or both).
    * The number of word embeddings that should be used.
    * The file paths to;
        * the file where the created data file should be stored,
        * the file where the trained model weights will be stored,
        * the file where the automatically combined training and test files should be stored,
        * the word embedding file,
        * the number of training sentences that should be extracted for each language,
        * the training sentences and their entity files for each language and
        * the test sentences and their entity files for each language.
    * The test and retrain options.
4. Upon running the file, the sentences of all languages will be combined into a new training set and a new data file will be created in which all the needed data is properly stored (if it does not yet exist). Then, a new language-consistent model will be trained (if it does not yet exist). The weights of the trained model are stored in the defined .h5 file.
5. If the test option is set to True, the newly trained model will be tested on the combined test sentences. The system outputs a precision, recall and F1-score.

#### LOREM
Now that we trained both the language-individual and -consistent sub-models, we can combine them to form the full LOREM model.
1. Open the file _LOREM.py_ and define the following variables:
    * The model types of the trained language-individual and -consistent models (CNN, LSTM or both).
    * The file paths to;
        * the language-individual and -consistent data files,
        * the language-individual and -consistent trained model files,
        * the test sentences and their entity files and
        * if the test language is not included in either the language-individual or language-consistent model, define the path to the test language word embeddings.
    * The options to;
        * test the model,
        * clean predictions for invalid tag sequences,
        * include the test language in the language-individual or language-consistent model if it is not included in these models.
2. Upon running the file, the predictions of the trained language-individual and -consistent models are computed and combined to end up with the final predicted tags for each word.
3. If the test option is set to True, LOREM will be tested on the defined test sentences. The system outputs a precision, recall and F1-score.

### Extracting relations
Once the full LOREM model is trained and tested, we can use it to extract relations from new sentences.
1. Make sure that the sentences are formatted correctly. The required format is described in section [Data format](#data-format).
2. Open the file _LOREM.py_ and in addition to the variables that were defined for testing LOREM, define the following variables:
    * The file path to the test file and the corresponding entity tags (the test_truth_file can remain empty).
    * The file path to the file where the predictions should be stored.
    * Set the export predictions option to True.
3. Upon running the file, the predictions are computed and extracted tuples are exported to the defined file.

## Data format
LOREM uses different files which should be formatted correctly for training, testing and extracting to work.

### Word embeddings
The word embeddings files should be .txt files which are formatted as follows:
> <pre> word1   embedding_values1 <br/> word2   embedding_values2 <br/> ...     ...</pre>

So for example:
> <pre> the     0.538   0.595   ... 0.129 <br/> guitar  0.218   0.005   ... 0.931 <br/> ...     ...     ...     ... ...</pre>

### Training and test sentences
Both training and test sentences should be formatted as JSON files. The training sentences are formatted as follows:
> <pre>{"senid": ["1"], "language": ["en"] tokens": ["word1", "word2", ...], "tags": ["tag1", "tag2", ...]} <br/>{"senid": ["2"], "language": ["fr"] tokens": ["word1", "word2", ...], "tags": ["tag1", "tag2", ...]}</pre>

So for example:
> <pre>{"senid": ["28998287"], "language": ["en"], "tokens": ["Ten", "is", "the", "debut", "album", "of", "Pearl", "Jam", ",", "released", "in", "1991", "."], "tags": ["O", "R-S", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}</pre>

The test sentences are formatted exactly the same. If we don't want to predict test scores and only want to extract relations, the "tags" values can be left out.

The corresponding entities are formatted in a highly similar fashion. For the previous example, these should be:
> <pre>{"senid": ["28998287"], "language": ["en"], "tokens": ["Ten", "is", "the", "debut", "album", "of", "Pearl", "Jam", ",", "released", "in", "1991", "."], "tags": ["E1-S", "O", "E2-B", "E2-I", "E2-I", "E2-I", "E2-I", "E2-E", "O", "O", "O", "O", "O"]}</pre>

### Exported predictions
The exported predictions are formatted as follows:
> <pre>Sent: The first sentence from which the relation is extracted . <br/>Pred: &lt Entity1 , Relation , Entity2 &gt <br/><br/>Sent: The second sentence from which the relation is extracted . <br/>Pred: &lt Entity1 , Relation , Entity2 &gt </pre>

For example:
> <pre>Sent: At least 8 schoolchildren were killed and at least 15 people were wounded when a deranged man burst into an elementary school near Osaka and began stabbing students and teachers with a kitchen knife . <br/>Pred: &lt a deranged man , began stabbing , students and teachers &gt</pre>

## References
The basis of LOREM is the NST model. The code that is published by its writers forms the basis of the repository you are currently visiting. The NST model can be found in:
> Shengbin Jia, Yang Xiang, and Xiaojun Chen. Supervised Neural Models Revitalize the Open Relation Extraction. CoRR, abs/1809.09408, 2018.

LOREM incorporates the idea of language-consistent relation extraction as was presented for the closed domain by:
> Xiaozhi Wang, Xu Han, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Adversarial Multi-lingual Neural Relation Extraction. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1156–1166, 2018.

For our experiments, we used English data from:
> Lei Cui, Furu Wei, and Ming Zhou. Neural Open Information Extraction. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, volume2, pages 407–413. Association for Computational Linguistics, 2018.

Open Information Extraction data for other languages was published by:
> Manaal Faruqui and Shankar Kumar. Multilingual Open Relation Extraction Using Cross-lingual Projection. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1351–1356. Association for Computational Linguistics, 2015.

We utilize pre-trained monolingual and multilingual word embeddings from respectively:
> https://fasttext.cc/docs/en/crawl-vectors.html <br/> https://fasttext.cc/docs/en/aligned-vectors.html
