"""Metrics to assess performance on nested NER task given prediction
These metrics are still part of a work in progress, so they have not yet 
been officially published as a library.
"""

from nested_entities_utils import get_nestings

def standard_metric(entities):
  tp = 0
  fn = 0
  fp = 0
  support = 0

  for sent in entities:

    p = sent["pred"]
    g = sent["real"]

    

    for entity in p: 
        if entity in g: 
            tp+=1
        if entity not in g:
            fp+=1

    for entity in g:
        support+=1
        if entity not in p:
            fn+=1
  
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = (2*precision*recall)/(precision+recall)

  return precision, recall, f1, support

def nesting_metric(entities):
  
  nesting_tp = 0
  nesting_fn = 0
  nesting_fp = 0
  support = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      support+=1
      if nesting in pred_nestings:
        nesting_tp+=1
      else:
        nesting_fn+=1

    for nesting in pred_nestings:
      if nesting not in test_nestings:
        nesting_fp+=1
  nesting_precision = nesting_tp/(nesting_tp+nesting_fp)
  nesting_recall = nesting_tp/(nesting_tp+nesting_fn)
  nesting_f1 = 2*(nesting_precision*nesting_recall)/(nesting_precision+nesting_recall)
  return nesting_precision, nesting_recall, nesting_f1, support

def flat_metric(entities):
  
  flat_tp = 0
  flat_fn = 0
  flat_fp = 0
  support = 0
  total = 0
  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    pred_flat_entities = []
    for entity in sent["pred"]:
      is_nested = False
      for nesting in pred_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        pred_flat_entities.append(entity)
    

    test_nestings = get_nestings(sent["real"])
    test_flat_entities = []
    for entity in sent["real"]:
   
      is_nested = False
      for nesting in test_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        test_flat_entities.append(entity)

    

    for entity in test_flat_entities:
      support+=1
      if entity in sent["pred"]:
        flat_tp+=1
      else:
        flat_fn+=1

    for entity in pred_flat_entities:
      if entity not in sent["real"]:
        flat_fp+=1

  flat_precision = flat_tp/(flat_tp+flat_fp)
  flat_recall = flat_tp/(flat_tp+flat_fn)
  flat_f1 = 2*(flat_precision*flat_recall)/(flat_precision+flat_recall)
  return flat_precision, flat_recall, flat_f1, support


def outer_metric(entities):
  
  outer_tp = 0
  outer_fn = 0
  outer_fp = 0
  support = 0
  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      support+=1
      if nesting[0] in sent["pred"]:
        outer_tp+=1
      else:
        outer_fn+=1

    for nesting in pred_nestings:
      if nesting[0] not in sent["real"]:
        outer_fp+=1
  
  outer_precision = outer_tp/(outer_tp+outer_fp)
  outer_recall = outer_tp/(outer_tp+outer_fn)
  outer_f1 = 2*(outer_precision*outer_recall)/(outer_precision+outer_recall)
  return outer_precision, outer_recall, outer_f1, support


def inner_metric(entities):
  support = 0
  inner_tp = 0
  inner_fn = 0
  inner_fp = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])

    for nesting in test_nestings:
      for entity in nesting[1:]:
        support+=1
        if entity in sent["pred"]:
          inner_tp+=1
        else:
          inner_fn+=1

    for nesting in pred_nestings:
      for entity in nesting[1:]:
        if entity not in sent["real"]:
          inner_fp+=1

  inner_precision = inner_tp/(inner_tp+inner_fp)
  inner_recall = inner_tp/(inner_tp+inner_fn)
  inner_f1 = 2*(inner_precision*inner_recall)/(inner_precision+inner_recall)
  return inner_precision, inner_recall, inner_f1, support



def nested_metric(entities):
  
  nested_tp = 0
  nested_fn = 0
  nested_fp = 0
  support = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      for entity in nesting:
        support+=1
        if entity in sent["pred"]:
          nested_tp+=1
        else:
          nested_fn+=1

    for nesting in pred_nestings:
      for entity in nesting:
        if entity not in sent["real"]:
          nested_fp+=1
    
  nested_precision = nested_tp/(nested_tp+nested_fp)
  nested_recall = nested_tp/(nested_tp+nested_fn)
  nested_f1 = 2*(nested_precision*nested_recall)/(nested_precision+nested_recall)
  return nested_precision, nested_recall, nested_f1, support