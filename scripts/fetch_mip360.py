"""Fetch complete selected scene scales/SfM from the official ZIP by byte range."""
import concurrent.futures,hashlib,io,json,struct,time,urllib.request,zipfile,zlib
from pathlib import Path
import argparse

URL='https://storage.googleapis.com/gresearch/refraw360/360_v2.zip'
PARSER=argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'workspace/data/mip360')
ROOT=PARSER.parse_args().output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
EXPECTED={r['path']:r for r in json.loads((Path(__file__).resolve().parents[1]/'results/gs3d/data-manifest.json').read_text())['files']}
SCALES={'bonsai':'images_2','bicycle':'images_4','garden':'images_4'}
with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':'bytes=0-3'}),timeout=60) as r:
    assert r.status==206
    SIZE=int(r.headers['Content-Range'].split('/')[-1]);ETAG=r.headers['ETag']
def get(start,end):
    end=min(end,SIZE-1)
    if start>end:return b''
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','If-Match':ETAG})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=90) as r:
                assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
                blob=r.read()
            assert len(blob)==end-start+1
            return blob
        except Exception:
            if attempt==2:raise
            time.sleep(1)
class Remote(io.RawIOBase):
    def __init__(self):self.pos=0
    def seekable(self):return True
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        n=SIZE-self.pos if n<0 else min(n,SIZE-self.pos)
        assert n<64*1024*1024, n
        b=get(self.pos,self.pos+n-1);self.pos+=len(b);return b
with zipfile.ZipFile(Remote()) as archive:
    selected=[]
    for info in archive.infolist():
        parts=Path(info.filename).parts
        assert '..' not in parts and not Path(info.filename).is_absolute()
        index=next((i for i in (0,1) if len(parts)>i and parts[i] in SCALES),None)
        if index is None or info.is_dir():continue
        parts=parts[index:];scene=parts[0]
        if len(parts)>1 and (parts[1] in (SCALES[scene],'sparse') or parts[1]=='poses_bounds.npy'):
            selected.append((info,Path(*parts)))
assert all(any(p.parts[0]==s and p.parts[1]==scale for _,p in selected) for s,scale in SCALES.items())
print(json.dumps({'archive_bytes':SIZE,'etag':ETAG,'selected_files':len(selected),'selected_uncompressed_bytes':sum(x.file_size for x,_ in selected),'scenes':SCALES}),flush=True)
def fetch(item):
    info,name=item;dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    content=dest.read_bytes() if dest.is_file() else None
    if content is None or len(content)!=info.file_size or zlib.crc32(content)!=info.CRC:
        extra=30+len(info.filename.encode('utf8'))+len(info.extra)+512
        blob=get(info.header_offset,info.header_offset+extra+info.compress_size-1)
        header=struct.unpack('<4s5H3I2H',blob[:30]);assert header[0]==b'PK\x03\x04'
        start=30+header[-2]+header[-1]
        compressed=blob[start:start+info.compress_size]
        if len(compressed)!=info.compress_size:
            compressed=get(info.header_offset+start,info.header_offset+start+info.compress_size-1)
        if info.compress_type==zipfile.ZIP_DEFLATED:content=zlib.decompress(compressed,-15)
        elif info.compress_type==zipfile.ZIP_STORED:content=compressed
        else:raise ValueError(info.compress_type)
        assert len(content)==info.file_size and zlib.crc32(content)==info.CRC,str(name)
        temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(content);temp.replace(dest)
    record={'path':str(name),'bytes':len(content),'crc32':f'{info.CRC:08x}','sha256':hashlib.sha256(content).hexdigest()}
    if str(name) not in EXPECTED or record['sha256']!=EXPECTED[str(name)]['sha256']:raise ValueError('public data bytes differ: '+str(name))
    return record
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    for record in executor.map(fetch,selected):
        records.append(record)
        if len(records)%25==0:print('completed',len(records),'of',len(selected),flush=True)
(ROOT/'PUBLIC_SOURCE.json').write_text(json.dumps({'archive_url':URL,'archive_bytes':SIZE,'archive_etag':ETAG,'scenes':SCALES,'scope':'All images at original official evaluation scale and full SfM for each selected scene; train/test split remains upstream --eval. Full-resolution unused image scales not duplicated.','files':records},indent=2))
print('COMPLETE',len(records),sum(x['bytes'] for x in records),flush=True)
