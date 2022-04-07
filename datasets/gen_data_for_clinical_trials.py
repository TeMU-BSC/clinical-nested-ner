import os 
import glob 
import spacy
import re
from tqdm import tqdm

def get_files(path):

  # Plain text files.
  text_files = glob.glob(f"{path}/abstracts/*.txt") + glob.glob(f"{path}/eudract/*.txt")
  text_files.sort()

  # Annotation files.
  ann_files = glob.glob(f"{path}/abstracts/*.ann") + glob.glob(f"{path}/eudract/*.ann")
  ann_files.sort()

  # Verifying that each text file has an associated annotation file.
  for t, a in zip(text_files, ann_files):
      assert(t.split('/')[-1].split('.')[0] == a.split('/')[-1].split('.')[0])
    
  return text_files, ann_files

def get_content(text_files, ann_files):
  
  texts, annotations = [], []

  for (text_file, ann_file) in tqdm(zip(text_files, ann_files), total=len(text_files)):
    # We open the file and save the content of the plain text.
    text_content = open(text_file, 'r').read()
    texts.append((text_file, text_content))

    # We open the file and save the content of the annotation.
    ann_content = open(ann_file, 'r').read() 
    annotations.append((ann_file, ann_content))
    
  return texts, annotations

def get_entity_annotations(annotations):
  total_entities = []
  entities_per_annotation = []

  for ann in annotations:

    entities = []

    for line in ann[1].splitlines():
      if line.startswith('T'):
        
        entity = {}

        line_info = line.split('\t')
        entity['text'] = line_info[2]
        entity['label'] = line_info[1].split()[0]
        entity['start_index'] = int(line_info[1].split()[1])
        entity['end_index'] = int(line_info[1].split()[2])
        total_entities.append(entity)
        entities.append(entity)

    entities_per_annotation.append(entities)
  return entities_per_annotation, total_entities

def check_inconsistencies(texts, annotations):
  # Check if the indexes in annotations match with the indexes in the plain text
  bad_entities = 0

  valid_annotations = []

  for text, entities in zip(texts, annotations):
    valid_entities = []
    text_content = text[1]
    
    for entity in entities:
     
      if entity['text']!=text_content[entity["start_index"]:entity["end_index"]]:
        bad_entities += 1
        print('Index error in entity {}'.format(entity['text']))
        print(f'Annotated indexes: Start_index = {entity["start_index"]} ; Start_index = {entity["end_index"]} ')
        print("Real text found using the above indexes: {}".format(text_content[entity["start_index"]:entity["end_index"]]))
      
      else:
        valid_entities.append(entity)

    valid_annotations.append(valid_entities)
  return valid_annotations, bad_entities


def get_entity_types(annotations):
    entity_types = [entity['label'] for anno in annotations for entity in anno]
    return list(set(entity_types))


def gen_nested_ner_format(tokenizer, texts, annotations, entity_types, output_filename):
    total_entities = []
    output_file = open(output_filename, 'w')
    entity_types_map = {x: i for i, x in enumerate(entity_types)}
    
    for text, annotation in tqdm(zip(texts, annotations), total = len(texts)):
        #output_file.write(text[0]+'\n')
        doc = tokenizer(text[1])
        entities = sorted(annotation, key = lambda entity: entity["start_index"])
        
        
        annotated_entities = []
        for sent in doc.sents:
            
            for token in sent:
                if not token.text or '\n' in token.text or '\t' in token.text or token.text.strip()=='':
                    continue
                
                token_start = token.idx
                token_end = token.idx + len(token.text)
                token_labels = ['O']*len(entity_types_map)

                for entity in entities:

                    if token_start==entity['start_index']:
                        token_labels[entity_types_map[entity['label']]] = f"B-{entity['label']}"
                        total_entities.append(entity)
                        annotated_entities.append(entity)
                        
                    if token_start > entity['start_index'] and token_end <=entity['end_index']:
                        token_labels[entity_types_map[entity['label']]] = f"I-{entity['label']}"
                
                
                output_file.write(f"{token.text} {' '.join(token_labels)}\n")
                 
            output_file.write("\n")

    output_file.close()
    return total_entities

def create_mlc_data(path, partition, entity_types):
    text = open(path, 'r').read()
    text = re.sub(r'\n\s*\n', '\n\n', text)

    if not os.path.exists(f'formatted_data/MLC/'):
          os.mkdir(f'formatted_data/MLC/')

    if not os.path.exists(f'formatted_data/MLC/clinical_trials'):
          os.mkdir(f'formatted_data/MLC/clinical_trials')

    for entity in entity_types:
        if not os.path.exists(f'formatted_data/MLC/clinical_trials/{entity}'):
          os.mkdir(f'formatted_data/MLC/clinical_trials/{entity}')

        f_out = open(f'formatted_data/MLC/clinical_trials/{entity}/{entity}_{partition}.iob2', 'w')
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line!='':
                original_entities = line.split()[1:]
                tag = 'O'
                for val in original_entities:
                    if val in (f'B-{entity}',f'I-{entity}'):
                        tag = val
                        break
                        
                f_out.write(f"{line.split()[0]} {tag}\n")
            else:
                if i!=len(lines)-1: f_out.write(line+'\n')
        f_out.close()
    
if __name__=='__main__':
    actual_path = os.getcwd()

    
    train_path = os.path.join(actual_path, "raw_data/clinical_trials/train")
    dev_path = os.path.join(actual_path, "raw_data/clinical_trials/dev")
    test_path = os.path.join(actual_path, "raw_data/clinical_trials/test")
   

    train_text_files, train_ann_files = get_files(train_path)
    dev_text_files, dev_ann_files = get_files(dev_path)
    test_text_files, test_ann_files = get_files(test_path)

    train_texts, train_annotations = get_content(train_text_files, train_ann_files)
    dev_texts, dev_annotations = get_content(dev_text_files, dev_ann_files)
    test_texts, test_annotations = get_content(test_text_files, test_ann_files)

    train_annotations, train_entities = get_entity_annotations(train_annotations)
    print("Total entities in train partition: {}".format(len(train_entities)))

    dev_annotations, dev_entities = get_entity_annotations(dev_annotations)
    print("Total entities in dev partition: {}".format(len(dev_entities)))

    test_annotations, test_entities = get_entity_annotations(test_annotations)
    print("Total entities in test partition: {}".format(len(test_entities)))

    
    train_annotations, train_errors = check_inconsistencies(train_texts, train_annotations)
    dev_annotations, dev_errors = check_inconsistencies(dev_texts, dev_annotations)
    test_annotations, test_errors = check_inconsistencies(test_texts, test_annotations)

    print("Incorrectly formatted annotations: {}\n".format(train_errors + dev_errors + test_errors))

    entity_types = get_entity_types(train_annotations + dev_annotations + test_annotations)

    print("Entity types:\n{}".format('\n'.join(entity_types)))

    tokenizer = spacy.load('es_core_news_sm')

    train_entities = gen_nested_ner_format(tokenizer, train_texts, train_annotations, entity_types, 'formatted_data/clinical_trials/train.iob2')
    print("Total entities in train partition: {}".format(len(train_entities)))

    dev_entities = gen_nested_ner_format(tokenizer, dev_texts, dev_annotations, entity_types, 'formatted_data/clinical_trials/dev.iob2')
    print("Total entities in dev partition: {}".format(len(dev_entities)))

    test_entities = gen_nested_ner_format(tokenizer, test_texts, test_annotations, entity_types, 'formatted_data/clinical_trials/test.iob2')
    print("Total entities in test partition: {}".format(len(test_entities)))

    create_mlc_data('formatted_data/clinical_trials/train.iob2', 'train', entity_types)
    create_mlc_data('formatted_data/clinical_trials/dev.iob2', 'dev', entity_types)
    create_mlc_data('formatted_data/clinical_trials/test.iob2', 'test', entity_types)