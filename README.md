# Political Classifier
BERT-based binary classifiers for web page titles in English, Polish and Dutch (multilingual), and a separate one for English only.

The models are large files, so they use git LFS. Make sure to clone with git LFS or you will only get pointers to these files, not the files themselves. Additionally, the model files are compressed tgz files; please decompress the downloaded tgz, which will result in a directory containing all necessary files. 

The run_model.py file takes 3 arguments when run from the command line, the input file location, the output file location, and the model directory. You will need to go through the run_model.py file to replace the existing file paths with your own.

The input file should have 2 variables, min_id and title, and the output predictions will have the min_id and the prediction (1/0) - not the title.
