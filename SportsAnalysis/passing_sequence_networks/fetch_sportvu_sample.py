"""Fetch and verify the pinned three-game SportVU sample; no repository code is executed."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,shutil
import requests
COMMIT='5c9ee6e1387e052c19ab2d22e12120c2210cbe54'
FILES=[('01.04.2016.CHA.at.GSW.7z','0021500520','651800deba16e3d2253ff4c6e4cfcec2ab48c058','0ff6559d0a60f9f807bf226e28d5a5ee30a23229'),('01.05.2016.GSW.at.LAL.7z','0021500524','0166b9b750b39e19d6b9e7aa8ddfefff76f2c9e5','1a9ce8588fff709650058e5a496277939901f146'),('01.08.2016.GSW.at.POR.7z','0021500548','a9365ea81bf786c32a6183f2373fd0e1b2040e64','6ae702997ef002a7003c4fbdd2e8c3f3b2945fd2')]
def blob_sha(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dest',type=Path,default=Path(__file__).resolve().parent/'sportvu_sequence_data/raw');args=p.parse_args();args.dest.mkdir(exist_ok=True,parents=True);tar=shutil.which('bsdtar')
 if not tar:raise RuntimeError('Install libarchive/bsdtar to extract7z archives')
 records=[]
 for archive,gid,expected_archive,expected_events in FILES:
  for remote,local,expected in [('data/'+archive,archive,expected_archive),('data/events/'+gid+'.csv',gid+'_events.csv',expected_events)]:
   dest=args.dest/local;url='https://raw.githubusercontent.com/sealneaward/nba-movement-data/'+COMMIT+'/'+remote
   if not dest.exists():r=requests.get(url,timeout=60);r.raise_for_status();dest.write_bytes(r.content)
   b=dest.read_bytes()
   if blob_sha(b)!=expected:raise ValueError('Git blob integrity check failed: '+local)
   records.append(dict(file=local,url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=expected))
  dest=args.dest/(gid+'.json')
  if not dest.exists():
   members=subprocess.check_output([tar,'-tf',str(args.dest/archive)],text=True).splitlines()
   if members!=[gid+'.json']:raise ValueError('Unexpected archive members')
   with dest.open('wb') as handle:subprocess.run([tar,'-xOf',str(args.dest/archive),gid+'.json'],stdout=handle,check=True)
  print('Verified',archive,'and matching play-by-play',flush=True)
 (args.dest/'verified_sources.json').write_text(json.dumps(dict(repository='https://github.com/sealneaward/nba-movement-data',commit=COMMIT,files=records),indent=2))
if __name__=='__main__':main()
