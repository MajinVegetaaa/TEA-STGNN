"""Audit newly supplied graduation and STAAR files without modifying model data."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; raw=ROOT/'data/raw'
m=pd.read_csv(ROOT/'data/processed/master_school_year.csv',dtype={'campus_id':str})
records=[]
for folder in ['12_graduation','13_staar_eoc']:
 for p in sorted((raw/folder).rglob('*')):
  if not p.is_file() or 'reference_only' in p.parts or p.suffix not in ['.xlsx','.xls','.csv','.dat']:continue
  y=int(p.parent.name) if folder=='13_staar_eoc' else int(re.search(r'20\d\d',p.name)[0]); r={'file':str(p.relative_to(raw)),'filename_year':y,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  if p.suffix=='.xlsx':
   b=pd.ExcelFile(p);d=pd.read_excel(b,sheet_name=b.sheet_names[-1],dtype=str).fillna('');r['data_sheet']=b.sheet_names[-1]
  elif p.suffix=='.xls':d=pd.read_html(p,keep_default_na=False)[0].astype(str)
  else:d=pd.read_csv(p,header=1 if y==2024 else 0,dtype=str).fillna('')
  d.CAMPUS=d.CAMPUS.str.strip().str.lstrip("'").str.zfill(9)
  r.update(rows=len(d),columns=len(d.columns),invalid_ids=int((~d.CAMPUS.str.fullmatch(r'\d{9}')).sum()))
  cohort=set(m.loc[m.year==y,'campus_id']);r.update(cohort_size=len(cohort),cohort_matched=len(cohort&set(d.CAMPUS)))
  if folder=='12_graduation':
   keys=['CAMPUS','CALC_FOR_STATE_ACCT'];r['calculation_variants']=sorted(d.CALC_FOR_STATE_ACCT.unique().tolist())
   if 'AEA' in d:keys.append('AEA')
   r['duplicate_full_keys']=int(d.duplicated(keys).sum());r['duplicate_campus_rows']=int(d.CAMPUS.duplicated().sum())
   r['required_fields_present']=all(c in d for c in ['CAMP_ALLD','CAMP_ALLN_GRAD','CAMP_ALLR_GRAD'])
   v=pd.to_numeric(d.CAMP_ALLR_GRAD,errors='coerce');r['graduation_rates_outside_range_excluding_negative_masks']=int((v>100).sum())
   r['graduation_rate_tokens']=sorted(set(d.loc[v.isna() | (v<0),'CAMP_ALLR_GRAD']))
  else:
   r['embedded_rate_years']=sorted({int(a[1])+2000 for c in d for a in [re.search(r'(\d\d)[RDN]$',c)] if a})
   r['filename_year_present']=y in r['embedded_rate_years'];r['duplicate_campus_rows']=int(d.CAMPUS.duplicated().sum())
   rates=d[[c for c in d if re.search(r'\d\dR$',c)]].apply(pd.to_numeric,errors='coerce')
   r['numeric_rates_over_100']=int((rates>100).sum().sum());r['negative_tokens']=sorted(set(rates.where(rates<0).stack().tolist()))
  records.append(r)
for r in records:r['identical_other_files']=[o['file'] for o in records if o['sha256']==r['sha256'] and o['file']!=r['file']]
out=ROOT/'results/source_audit/staar_graduation_2026-11-16.json';out.write_text(json.dumps(records,indent=2))
for r in records:print(r['file'],{k:v for k,v in r.items() if k not in ['file','sha256','negative_tokens']})
