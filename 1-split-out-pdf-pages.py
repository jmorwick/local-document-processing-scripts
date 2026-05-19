# before this script was run:
# cidni --dataservice ./cidnidb know -r PDFs
# cidni --dataservice ./cidnidb list > original-pdf-ids.txt

from pypdf import PdfReader, PdfWriter
from io import BytesIO
import cidnilib
from cidnilib import FileBasedDataService as FBDS
from cidnilib import PickleFileBasedDataService as PFBDS
from cidnilib import InMemoryKnowledgeService as IMKS

ds = FBDS('./cidnidb')
kds = PFBDS('./cidnidb', levels=0)
ks = IMKS(kds)

with open("original-pdf-ids.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  
        cid = ds.decode(line)
        
        
        # retrieving annotations (example)
        #print('there are', len(list(ks.inquire(subject=ds.encode(cid)))), 'pages')
        #for page in ks.inquire(subject=ds.encode(cid)):
        #  kid, _ = ks.believe(page[0], page[1], page[2])
        #  print('page:',list(ks.inquire(subject=ds.encode(kid))))
        
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
            
            

kds.flush()
