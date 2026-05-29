from io import BytesIO
import cidnilib
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
import json
import sys
from collections import defaultdict


if len(sys.argv) < 3:
   print("usage: python script pdf_list_cid data-matrix-file k", file=sys.stderr)

pdf_list_cid = sys.argv[1]
data_file_name = sys.argv[2]
k = int(sys.argv[3])


matrix = np.loadtxt(data_file_name, delimiter=",")

print(f"doing {k}-NN on {data_file_name} with shape {matrix.shape}", file=sys.stderr)

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)


document_cids = []

for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        document_cids.append(cid)
        
def get_property(cid, property):
    if type(cid) == bytes: cid = ds.encode(cid) 
    try: return next(ks.inquire(cid, property))[2]
    except: return None
        

def max_weighted_vote(doc_similarities):
    tallies = defaultdict(lambda : 0)
    for cid, conf in doc_similarities:
      cl = get_property(cid, 'assigned-class')
      tallies[cl] += conf
    print('votes',tallies, file=sys.stderr)
    chosen_class = max(tallies, key=tallies.get)
    return chosen_class, tallies[chosen_class]/k


predictions = []
print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
for doc_cid in document_cids:
    print('\n\n processing pdf: ', ds.encode(doc_cid), file=sys.stderr)
    print(f"Progress: {(len(predictions)/len(document_cids)):.2%}\n\n", file=sys.stderr)
    row = matrix[r]
    others = []
    for c in range(len(row)):
        if doc_cid == document_cids[c]: continue
        others.append((document_cids[c],row[c]))
    others = sorted(others, key=lambda x: x[1], reverse=True)
    predicted_class, predicted_confidence = max_weighted_vote(others[0:k])
    predictions.append((ds.encode(doc_cid), get_property(doc_cid, 'assigned-class'), predicted_class, predicted_confidence))
    print(','.join(map(str,predictions[-1])))
    r += 1
correct = 0
for prediction in predictions:
    if prediction[1] == prediction[2]: correct+=1

print('accuracy:',correct/len(predictions), file=sys.stderr)

