from io import BytesIO
import cidnilib
import numpy as np
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS
import PIL
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


if len(sys.argv) < 4:
   print("usage: python script pdf_list_cid ocr_model embedding_model", file=sys.stderr)
   sys.exit()
   
num_threads = 14
pdf_list_cid = sys.argv[1]
ocr_model_name = sys.argv[2]
embedding_model_name = sys.argv[3]

print('using embeddings from',embedding_model_name,'derived from OCR from',ocr_model_name,'to compare document similarity', file=sys.stderr)

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
  
def get_pages(cid):
    embeddings = []
    page_cids = dict()
    for _, _, page_cid in ks.inquire(ds.encode(cid), 'CONTAINS'):
        kid, _ = ks.believe(ds.encode(cid), 'CONTAINS', page_cid)
        page_cids[get_property(kid, 'PAGE')] = page_cid
    for page_num in sorted(page_cids):
        yield page_cids[page_num]


def get_page_embeddings(page_cid):
    for _, _, emcid in ks.inquire(subject=page_cid, property='EMBEDDING'):
      okid, _ = ks.believe(page_cid, 'EMBEDDING', emcid)
      for _, _, cmodel in ks.inquire(subject=ds.encode(okid), property='MODEL'):
        if cmodel == embedding_model_name: 
            return np.array(json.loads(ds.recall(emcid))[0])


print('Total documents to process:',len(document_cids), file=sys.stderr)
r = 0
all_pages = []
first_pages = set()
for cid1 in document_cids: 
    print('\n\n loading pdf pages: ', ds.encode(cid1), file=sys.stderr)
    print(f"Progress: {(r/len(document_cids)):.2%}\n\n", file=sys.stderr)

    pages = get_pages(cid1)
    first_pages.add(len(all_pages))
    all_pages.extend(pages)
    r += 1

chunk_size = len(all_pages)//num_threads

def worker(ti):
  output_chunk = []

  for i in range(chunk_size +(0 if ti < num_threads-1 else len(all_pages)%num_threads)): 
    cid1 = all_pages[ti*chunk_size + i] 
    print(f"\n\n worker {ti} loading pdf page: {cid1}", file=sys.stderr)
    print(f"Progress: {(i/(len(all_pages)/num_threads)):.2%}\n\n", file=sys.stderr)

    d1em = get_page_embeddings(cid1)
    row = []
    for cid2 in all_pages:
        d2em = get_page_embeddings(cid2)
        if d1em is None or d2em is None:
            dist = 0
        else: 
            dist = np.dot(d1em, d2em)
        row.append(dist)
    output_chunk.append(row)

  return ti, output_chunk


if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = [ executor.submit(worker, ti) for ti in range(num_threads) ]

        output_chunks = [None] * num_threads

        for future in as_completed(futures):
            ti, chunk = future.result()
            output_chunks[ti] = chunk
            print(f"worker {ti} complete", file=sys.stderr)

    # write to csv file
    m = []
    for chunk in output_chunks:
      m.extend(chunk)

    np.savetxt(sys.stdout, m, delimiter=",")    
