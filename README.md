# NLP---Spacy

This Repo consists of a file spacyBasics.py, which contains some basic functionalities that are provided by the pretrained models in spacy.


#########################################################################################################################################

All the files other than spacyBasics.py are used to train a model for Named Entity Recognition (NER) using a pretrained model in spacy.

Steps :-

NER involves recognizing certain specific entities from a sentence.
The ner_dataset.csv file and ner_dataset.tsv files are the same and they are tagged using BIO tagging scheme also BILOU tags can be used. The end of sentence in these files are markes using a row for full stop.
Some examples of tags can be "B-org", "I-org" for Beginning and Inside or organization name respectively.
Run the spacyExample.py file by commenting lines between 83 and 183. It would use this tsv file and generate a "ner_dataset.json" file which would help us to properly create the training data in a format of list that the spacy would use.
Now uncomment and go through the code once. Change the directory where you would save the model.
Change any parameters if you wish.
Run the code and see the output for the testing text. Also, try it with other test text as inputs.


Use spacy version 2.2.0
Also download the "en_core_web_lg" model of spacy
