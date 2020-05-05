import random
import pickle
import os
import argparse
import plac
import sys
import logging
import json
import spacy
from spacy.util import minibatch, compounding
from pathlib import Path

# -----------------------------------------------------------------------------------------------------------------------------------


## Code to convert tsv file "ner_dataset.tsv" into json file "ner_dataset.json" The JSON file contains the content and annotation in a proper format

def tsv_to_json_format(input_path, output_path, unknown_label):
    try:
        f = open(input_path, 'r')  # input file
        fp = open(output_path, 'w')  # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        s = ''
        start = 0
        for line in f:
            if line[0:len(line)-1] != '.\tO':
                #print(line)
                word, entity = line.split('\t')
                s += word+" "
                entity = entity[:len(entity)-1]
                if entity != unknown_label:
                    if len(entity) != 1:
                        d = {}
                        d['text'] = word
                        d['start'] = start
                        d['end'] = start+len(word)-1
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity] = []
                            label_dict[entity].append(d)
                start += len(word)+1
            else:
                data_dict['content'] = s
                s = ''
                label_list = []
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text'] != ''):
                            l = [ents, label_dict[ents][i]]
                            for j in range(i+1, len(label_dict[ents])):
                                if(label_dict[ents][i]['text'] == label_dict[ents][j]['text']):
                                    di = {}
                                    di['start'] = label_dict[ents][j]['start']
                                    di['end'] = label_dict[ents][j]['end']
                                    di['text'] = label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text'] = ''
                            label_list.append(l)

                for entities in label_list:
                    label = {}
                    label['label'] = [entities[0]]
                    label['points'] = entities[1:]
                    annotations.append(label)
                data_dict['annotation'] = annotations
                annotations = []
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict = {}
                start = 0
                label_dict = {}
    except Exception as e:
        logging.exception("Unable to process file" +
                          "\n" + "error = " + str(e))
        return None


tsv_to_json_format("ner_dataset.tsv", 'ner_dataset.json', 'abc')


@plac.annotations(input_file=("Input file", "option", "i", str), output_file=("Output file", "option", "o", str))
def main(input_file, output_file):
    try:
        training_data = []
        lines = []
        input_file = "ner_dataset.json"
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))

            training_data.append((text, {"entities": entities}))

        #print(training_data)
        ## This is the training_data format that is required by spacy to train the model. It is constructed from the data in the ner_dataset.json file

        with open('trainData.txt', 'w') as write:
            write.write(str(training_data))

    except Exception as e:
        logging.exception("Unable to process " +
                          input_file + "\n" + "error = " + str(e))
        return None


    ## The Large pretrained english model is loaded
    nlp = spacy.load('en_core_web_lg')
    ## Task Named Entity Recognizer is added into the pipeline
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')


    ## These are the labels that the entities in the text would be trained upon. These are tagged in the main training tsv file
    labels = ['I-geo', 'B-geo', 'I-art', 'B-art', 'B-tim', 'B-nat', 'B-eve', 'O',
                      'I-per', 'I-tim', 'I-nat', 'I-eve', 'B-per', 'I-org', 'B-gpe', 'B-org', 'I-gpe']
    for i in labels:
        ner.add_label(i)
    # Inititalizing optimizer
    optimizer = nlp.begin_training()


    ## Except NER all other tasks have been disabled
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']


    with nlp.disable_pipes(*other_pipes):  # only train NER
        TRAIN_DATA = training_data
        for itn in range(50): ## Try to change the number of epochs and run the model
            random.shuffle(TRAIN_DATA)
            losses = {}
            ## Try to change the size and run the model
            batches = minibatch(TRAIN_DATA, size=compounding(10., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                ## Updating the weights
                ## Try to change the drop rates and run the model
                nlp.update(texts, annotations, sgd=optimizer, drop=0.25, losses=losses)

            print('Losses', losses)


    # Save model
    ## Change the directory here to save the model
    output_dir = "C:\\Users\\138709\\Desktop\\Invoice data extraction\\NLTK and Spacy\\Spacy Test" 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = "nermodel"  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    
    ## Testing the NER model with a given text
    test_text = 'Gianni Infantino is the president of FIFA.'
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(test_text)

    ## Printing the named entities that the model has extracted
    for ent in doc2.ents:
        print(ent.label_, ent.text)

if __name__ == '__main__':
    plac.call(main)


