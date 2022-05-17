# Political Classifier
BERT-based binary classifiers for web page titles in English, Polish and Dutch (multilingual), and a separate one for English only.

The models are large files, so they use git LFS. Make sure to clone with git LFS or you will only get pointers to these files, not the files themselves. Additionally, the model files are compressed tgz files; please decompress the downloaded tgz, which will result in a directory containing all necessary files. 

The run_model.py file takes 3 arguments when run from the command line, the input file location, the output file location, and the model directory. You will need to go through the run_model.py file to replace the existing file paths with your own.

The input file should have 2 variables, min_id and title, and the output predictions will have the min_id and the prediction (1/0) - not the title.

If you wish to train a model using new data, the script to start with is classify_bert.py (if you have not created a training/validation/test split) or classify_bert_split.py (if you have your training, validation, and test data in separte CSV files). These scripts take 2 arguments (data file, model save location) or 3 arguments (train file, validation file, save location), respectively. These two scripts fine-tune the "bert_base_multilingual_cased" model from Google as a classifier on the provided data.

In order to automatically classify online news article titles as political or non-political in nature, we developed two neural binary classifiers, built on top of a large transformer-based language model, namely BERT (Devlin et al.,2019). Our first model is for classification of English-language news article titles, while the second model is able to identify political content in English, Dutch, and Polish. 

BERT is a deep transformer model pre-trained on huge amounts of unlabeled text using a masked-word prediction training objective. By training on this objective with such a large amount of data, BERT builds a powerful language model which can then be fine-tuned to successfully complete specific language-related tasks, such as question answering and text classification. By pretraining the model on unlabeled text, the model is essentially taught to understand the target language. Then by fine-tuning, the model learns how to complete certain specific tasks with its knowledge of the language.

Given the BERT is trained on huge amounts of web data, including news articles, we did not feel that additional pre-training of the BERT model was necessary for the target domain. We utilize the Google's ‘bert-base-uncased’ pre-trained model for English, and the 'bert-base-multilingual-cased' model for our multilingual data. We implement our models using Huggingface’s Transformers package for Python. We encode our tokenized news article titles using the pretrained BERT model. We then pass the BERT model’s final hidden layer to a linear softmax layer for binary-class prediction. 

Our English training dataset consists of 2887 news article titles which were manually labeled for political or non-political status by two trained annotators 234 coded article titles were set aside for model testing and the model was fine-tuned for four epochs with 10% of the remaining coded data set aside training validation. Similiary out multilingual consists of 4691 coded article titles drawn from US, Dutch, and Polish news sources. The classification results for both our English-only and our multilingual models are shown below. Additionally, two trained annotators validated the output from the classifiers, manually checking titles resulting from the classification. The annotators disagreed with roughly 5% of the output of the classifier, finding that the classifier may be more likely to label non-political content as political than vice versa.

English-only results:
![en_results](https://github.com/ercexpo/political_classification/blob/main/political_en_results.png "English results")

Multilingual results:
![ml_results](https://github.com/ercexpo/political_classification/blob/main/political_multi_results.png "Multilingual results")

Details on the coding process
Two annotators (the Principal Investigator and a trained PhD candidate) manually labeled the 2887 English and 4691 multilingual news article titles for whether they were political or non-political. The titles relevant to issues of public concern were considered as political. These not only include references to political figures, policies, elections, news events, and specific political events (e.g., impeachment inquiry, or the primaries), but also sociopolitical issues such as acusations of sexual assault, the regulation of large tech companies, issues related to racial, gender, sexual, ethnic, and religious idenitity, as well as crimes involving guns and shootings. 
