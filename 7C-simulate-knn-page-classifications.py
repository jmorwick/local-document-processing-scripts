from io import BytesIO
import cidnilib
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
import json
import sys


if len(sys.argv) < 6:
   print("usage: python script pdf_list_cid data-matrix-file k ocr_model embedding_model ", file=sys.stderr)

pdf_list_cid = sys.argv[1]
data_file_name = sys.argv[2]
k = int(sys.argv[3])
ocr_model_name = sys.argv[4]
embedding_model_name = sys.argv[5]


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
  
def get_parent(cid):
    if type(cid) is bytes: cid = ds.encode(cid)
    for parent_cid, _, _ in ks.inquire(None, 'CONTAINS', cid):
        return parent_cid

def get_pages(cid):
    if type(cid) is bytes: cid = ds.encode(cid)
    embeddings = []
    page_cids = dict()
    for _, _, page_cid in ks.inquire(cid, 'CONTAINS'):
        kid, _ = ks.believe(cid, 'CONTAINS', page_cid)
        page_cids[get_property(kid, 'PAGE')] = page_cid
    return page_cids


def get_page_embeddings(cid):
    for _, _, emcid in ks.inquire(subject=cid, property='EMBEDDING'):
        emkid, _ = ks.believe(cid, 'EMBEDDING', emcid)
        if get_property(emkid, 'MODEL') == embedding_model_name: 
            return np.array(json.loads(ds.recall(emcid))[0])
    
         
print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
all_pages = []
first_pages = set()
for cid1 in document_cids: 
    print('\n\n loading pdf pages: ', ds.encode(cid1), file=sys.stderr)
    print(f"Progress: {(r/len(document_cids)):.2%}\n\n", file=sys.stderr)

    pages = get_pages(cid1)
    if '1' not in pages: 
        print(f"ERROR: no first page (all pages: {pages}", file=sys.stderr)
        continue
    first_pages.add(pages['1'])
    all_pages.extend([pages[pnum] for pnum in sorted(pages)])
    r += 1
    
r = 0   


predictions = []
print('Total pages to process:',len(document_cids), file=sys.stderr)
r = 0
for cid in all_pages:
    print('\n\n processing page: ', cid, 'from doc: ', get_parent(cid),file=sys.stderr)
    print(f"Progress: {(len(predictions)/len(all_pages)):.2%}\n\n", file=sys.stderr)
    row = matrix[r]
    others = []
    toskip = get_pages(get_parent(cid)).values()
    for c in range(len(row)):
        if all_pages[c] not in toskip: 
            others.append((all_pages[c],row[c]))
    others = sorted(others, key=lambda x: x[1], reverse=True)
    predictions.append((cid, cid in first_pages, others[0][0] in first_pages, others[0][1]))
    print('top-page:',others[0][0], 'doc:',get_parent(others[0][0]), file=sys.stderr)
    print(predictions[-1][0]+','+str(predictions[-1][1])+','+str(predictions[-1][2])+','+str(float(predictions[-1][3])))
    print(predictions[-1][0]+','+str(predictions[-1][1])+','+str(predictions[-1][2])+','+str(float(predictions[-1][3])),file=sys.stderr)
    r += 1
correct = 0
for prediction in predictions:
    if prediction[1] == prediction[2]: correct+=1

print('accuracy:',correct/len(predictions), file=sys.stderr)

