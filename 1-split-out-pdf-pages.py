# before this script was run:
# cidni --dataservice ./cidnidb know -r PDFs
# cidni --dataservice ./cidnidb list > original-pdf-ids.txt
# cidni --dataservice ./cidnidb know original-pdf-ids.txt
# use resulting cid for pdf list throughout experiments


from pypdf import PdfReader, PdfWriter
from io import BytesIO
import cidnilib
import sys
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS


if len(sys.argv) < 2:
   print("usage: python script pdf_list_cid")

pdf_list_cid = sys.argv[1]

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)

document_cids = ds.recall(ds.decode(pdf_list_cid)).decode().split('\n')
print('Total documents to process:',len(document_cids))
processed = 0
for line in ds.recall(ds.decode(pdf_list_cid)).decode().split('\n'):
        processed += 1
        line = line.strip() 
        if not line: continue
        cid = ds.decode(line)   
        
        print('\n\n processing pdf: ', line)
        print(f"Progress: {(processed/len(document_cids)):.2%}\n\n")
        
        try:
            reader = PdfReader(ds.recall_stream(cid))
            pagenum = 0
            for page in reader.pages:
                pagenum += 1
                writer = PdfWriter()
                writer.add_page(page)
                output = BytesIO()
                writer.write(output)
                output.seek(0)
                pcid, s = ds.know_file(output)
                kid, _ = ks.believe(ds.encode(cid), 'CONTAINS', ds.encode(pcid))
                kid2, _ = ks.believe(ds.encode(kid), 'PAGE', str(pagenum))
                print('(',pagenum, s,')', ds.encode(cid),'--->', ds.encode(pcid))
        except:
          print('ERROR: could not read', ds.encode(cid))
            
            

kds.flush()   # move up to recover from interruptions more easily
