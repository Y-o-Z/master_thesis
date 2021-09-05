### Requirements:

python==3.7.0

allennlp==0.8.4 \
torch==1.3.1 \
transformers==2.2.2 \
sentencepiece==0.1.91 \
scikit-learn==0.20.0 \
python-Levenshtein==0.12.0 \
symspellpy==6.7.0


### Code overview:

run shell training interface: => \
python t_tempora.py 9


produce exercises: => \
1) adapt path to pretrained models in g_predict.py => def get_single_model(), def get_multi_model()  (download pretrained models from https://github.com/grammarly/gector) \
2) create a folder 'z_test' in the current directory to hold the generated exercises \
3) python g_predict.py -m "" -v "" -i <path_to_input_file> -exercises 1 

=> read the generated pickle files with the read_pickle() method from the t_helper.py file \
=> the pickle files will be empty if there are no errors of the corresponding type in the input file  (g-transformations overview in GECToR paper)

