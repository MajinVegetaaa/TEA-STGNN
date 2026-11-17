"""Profile staged November 2026 source additions; does not change model inputs."""
from pathlib import Path
import hashlib, json, re, zipfile
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
raw = ROOT / 'data/raw'
master = pd.read_csv(ROOT/'data/processed/master_school_year.csv', dtype={'campus_id':str})
def ids(s): return s.astype(str).str.strip().str.lstrip("'").str.zfill(9)
records=[]
for folder in ['08_ap','09_staff_tapr','10_finance_peims','11_ib']:
 for p in sorted((raw/folder).iterdir()):
  if p.suffix not in ['.xlsx','.xls','.csv','.zip']: continue
  rec={'file':str(p.relative_to(raw)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if folder in ['08_ap','11_ib']:
   year=int(re.search(r'20\d\d-(\d\d)',p.name)[1])+2000
   book=pd.ExcelFile(p); d=pd.read_excel(book,sheet_name=book.sheet_names[-1],dtype=str).fillna('')
   d['Campus']=ids(d.Campus); a=d[d.Group=='All Students']
   rec.update(year=year,rows=len(d),campuses=d.Campus.nunique(),duplicate_group_campus=int(d.duplicated(['Group','Campus']).sum()),all_students_rows=len(a),groups=sorted(d.Group.unique().tolist()))
   cohort=set(master.loc[master.year==year,'campus_id']);rec.update(cohort_size=len(cohort),cohort_matched=len(cohort & set(a.Campus)))
   rec['rate_out_of_range']={c:int(((v:=pd.to_numeric(d[c],errors='coerce')).notna() & ((v<0)|(v>100))).sum()) for c in d if 'Rate' in c or 'rate' in c}
  elif folder=='09_staff_tapr':
   year=int(re.search(r'20\d\d',p.name)[0])
   d=(pd.read_html(p,keep_default_na=False)[0].astype(str) if p.suffix=='.xls' else pd.read_csv(p,header=1 if year==2024 else 0,dtype=str)).fillna(''); d['CAMPUS']=ids(d.CAMPUS)
   rec.update(year=year,year_status='header-confirmed' if year==2024 else 'filename year; coded fields have no embedded year',rows=len(d),columns=len(d.columns),duplicate_campus=int(d.CAMPUS.duplicated().sum()),invalid_ids=int((~d.CAMPUS.str.fullmatch(r'\d{9}')).sum()))
   cohort=set(master.loc[master.year==year,'campus_id']);rec.update(cohort_size=len(cohort),cohort_matched=len(cohort & set(d.CAMPUS)))
   rec['core_fields']={c: {'present':c in d, 'nonnegative_numeric':int((pd.to_numeric(d[c],errors='coerce')>=0).sum()) if c in d else 0} for c in ['CPSTEXPA','CPSTTENA','CPSTKIDR','CPSTTOSA','CPSTTOFC']}
  else:
   with zipfile.ZipFile(p) as z:
    rec['crc_ok']=z.testzip() is None; rec['members']=z.namelist();years=set();n=0;units=set(); districts=set()
    with z.open(z.namelist()[0]) as f:
     for d in pd.read_csv(f,dtype=str,chunksize=100000):
      n+=len(d);years.update(d.FUNDYEAR.dropna());units.update(d.FIN_UNIT.dropna());districts.update(d.DISTRICT.dropna())
    rec.update(rows=n,fundyears=sorted(years),districts=len(districts),financial_units=len(units),columns=d.columns.tolist())
  records.append(rec)
out=ROOT/'results/source_audit/downloads_2026-11-16.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(records,indent=2))
for r in records: print({k:v for k,v in r.items() if k in ['file','rows','year','fundyears','cohort_matched','cohort_size','duplicate_group_campus','duplicate_campus','rate_out_of_range','crc_ok']})
