from collections import defaultdict

def mlc_to_joint_labeling(dataset, entity_types):
  for partition in ('train', 'dev', 'test'):
    output_file = open(f'{dataset}_joint_labeling_{partition}.iob2', 'w')
    my_dict = defaultdict(list)
    for i, entity_type in enumerate(entity_types):
      file_content = open(f'formatted_data/MLC/{dataset}/{entity_type}/{entity_type}_{partition}.iob2', 'r').read()
      for j, line in enumerate(file_content.splitlines()):
        if line == '':
          my_dict[j].append('EOS')
          continue
        data = line.split()
        token = data[0]
        label = data[1]
        if i == 0:
            my_dict[j].append(token)
            my_dict[j].append(label)
        else:
            my_dict[j].append(label)

    for k, v in my_dict.items():
          if v[0] == 'EOS':
              output_file.write("\n")
          else:
              new_array = v[1:]
              output_file.write(f"{v[0]} {'+'.join(new_array)}\n")
    output_file.close()
    
entity_types = ['Disease', 'Medication', 'Finding', 'Abbreviation', 'Procedure', 'Family_Member', 'Body_Part']
dataset = 'wl'

mlc_to_joint_labeling(dataset, entity_types)