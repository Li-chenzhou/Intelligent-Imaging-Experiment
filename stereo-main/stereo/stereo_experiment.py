"""Reproducible classical stereo benchmark. OpenCV implements BM/SGBM.
Run: python stereo_experiment.py --data ../all/data --out results
All disparity errors are in ORIGINAL image pixels, including half-resolution runs.
"""
from pathlib import Path
import sys
VENDOR=Path(__file__).resolve().parent/'vendor'
if VENDOR.is_dir(): sys.path.insert(0,str(VENDOR))
import argparse,csv,json,re,time,hashlib,platform,datetime
import numpy as np
import cv2

SEED=20260930
FULL=[('BM_b9','BM',9,0),('SGBM_b5_p32','SGBM',5,32),
      ('SGBM_b5_p128','SGBM',5,128),('SGBM_b9_p32','SGBM',9,32),
      ('SGBM_b9_p128','SGBM',9,128)]

def read_pfm(path):
    with open(path,'rb') as f:
        magic=f.readline().decode('ascii').strip()
        if magic not in ('Pf','PF'): raise ValueError('Invalid PFM magic')
        def header():
            line=f.readline()
            while line.startswith(b'#') or not line.strip(): line=f.readline()
            return line
        w,h=map(int,header().split()); scale=float(header()); c=3 if magic=='PF' else 1
        a=np.fromfile(f,dtype='<f4' if scale<0 else '>f4')
        if a.size!=w*h*c: raise ValueError('PFM length mismatch')
        # Middlebury disparity PFM: magnitude of scale is ignored; samples are pixels.
        # https://vision.middlebury.edu/stereo/submit3/upload-format.html
        return np.flipud(a.reshape((h,w,c) if c==3 else (h,w))).copy()

def calib(path):
    d={}
    for line in path.read_text().splitlines():
        if '=' not in line: continue
        k,v=line.split('=',1)
        if k.startswith('cam'): d[k]=np.array([float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',v)]).reshape(3,3).tolist()
        else:d[k]=float(v)
    return d

def read_gray(path):
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError(f'Cannot read {path}')
    return im

def matcher(kind,b,p2,nd):
    if kind=='BM':
        m=cv2.StereoBM_create(numDisparities=nd,blockSize=b)
        m.setMinDisparity(0);m.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        m.setPreFilterSize(9);m.setPreFilterCap(31);m.setTextureThreshold(10)
        m.setUniquenessRatio(10);m.setSpeckleWindowSize(100);m.setSpeckleRange(2);m.setDisp12MaxDiff(1)
        return m
    return cv2.StereoSGBM_create(minDisparity=0,numDisparities=nd,blockSize=b,
        P1=8*b*b,P2=p2*b*b,disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=10,
        speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM)

def metrics(pred,gt,c):
    valid=np.isfinite(gt)&(gt>=0); pv=np.isfinite(pred)&(pred>=0); both=valid&pv
    if not valid.any() or not both.any(): raise ValueError('No evaluable pixels')
    error=np.abs(pred[both].astype(np.float64)-gt[both].astype(np.float64))
    result={'gt_pixels':int(valid.sum()),'pred_pixels':int(both.sum()),
      'coverage':100*float(both.sum()/valid.sum()),'epe_valid':float(error.mean()),
      'rmse_valid':float(np.sqrt(np.mean(error**2)))}
    for th in [1,2,4]:
        fail=int((valid&~pv).sum()+np.count_nonzero(error>th))
        result[f'bad{th}_total']=100*fail/int(valid.sum());result[f'bad{th}_pixels']=fail
    denom=pred+c['doffs']; dz=gt+c['doffs']; mask=both&(denom>0)&(dz>0)
    z=cv2.multiply(np.ones_like(pred),float(c['baseline']*c['cam0'][0][0]))
    est=z[mask]/denom[mask]; true=z[mask]/dz[mask]
    result['depth_absrel_valid']=float(np.mean(np.abs(est-true)/true))
    return result

def aggregate(rows,out):
    keys=['bad1_total','bad2_total','bad4_total','coverage','epe_valid','rmse_valid','ms_median','depth_absrel_valid','roi_bad2','common_epe']
    summary={}
    for scale,method in sorted({(r['scale'],r['method']) for r in rows}):
        rr=[r for r in rows if r['scale']==scale and r['method']==method]
        groups=sorted({r['scene_group'] for r in rr})
        bygroup={g:{k:float(np.mean([r[k] for r in rr if r['scene_group']==g])) for k in keys} for g in groups}
        summary[f'{scale}_{method}']={'view_macro':{k:float(np.mean([r[k] for r in rr])) for k in keys},
          'scene_macro':{k:float(np.mean([v[k] for v in bygroup.values()])) for k in keys},
          'scene_sd':{k:float(np.std([v[k] for v in bygroup.values()],ddof=1)) for k in keys},
          'bygroup':bygroup,'micro_bad2':100*sum(r['bad2_pixels'] for r in rr)/sum(r['gt_pixels'] for r in rr)}
    a=summary['1.0_BM_b9']['bygroup'];b=summary['1.0_SGBM_b5_p32']['bygroup']
    dif=np.array([a[g]['bad2_total']-b[g]['bad2_total'] for g in a])
    rng=np.random.default_rng(SEED);boots=dif[rng.integers(0,len(dif),(10000,len(dif)))].mean(1)
    effects={'BM_minus_SGBM_bad2_pp':float(dif.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'positive_groups':int((dif>0).sum())}
    def av(m):return summary['1.0_'+m]['scene_macro']['bad2_total']
    y00,y01,y10,y11=[av(m) for m in ['SGBM_b5_p32','SGBM_b5_p128','SGBM_b9_p32','SGBM_b9_p128']]
    effects['window9_minus5_pp']=((y10+y11)-(y00+y01))/2
    effects['P2_128_minus32_pp']=((y01+y11)-(y00+y10))/2
    effects['difference_of_differences_pp']=(y11-y10)-(y01-y00)
    ordered=sorted([r for r in rows if r['scale']==1 and r['method']=='SGBM_b5_p32'],key=lambda r:r['bad2_total'])
    result={'summary':summary,'effects':effects,'selected':[ordered[i]['view'] for i in [0,len(ordered)//2,len(ordered)-1]]}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'all/data');p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    rng=np.random.default_rng(SEED);views=sorted(x for x in a.data.iterdir() if x.is_dir());rng.shuffle(views)
    rows=[];times=[];manifest=[];schedule=[]
    meta={'started':datetime.datetime.now().isoformat(),'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,'os':platform.platform(),'threads':cv2.getNumThreads(),'seed':SEED,'data':str(a.data.resolve()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.out/'environment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for i,v in enumerate(views):
        c=calib(v/'calib.txt');left=read_gray(v/'im0.png');right=read_gray(v/'im1.png');gt=read_pfm(v/'disp0.pfm');gtr=read_pfm(v/'disp1.pfm')
        assert left.shape==right.shape==gt.shape==gtr.shape
        mf={'view':v.name,'scene_group':re.sub(r'\d+$','',v.name),'shape':list(gt.shape),'gt_coverage':float(np.mean(np.isfinite(gt)&(gt>=0))*100),'ndisp':int(c['ndisp']),'baseline_mm':c['baseline'],'focal_px':c['cam0'][0][0], 'hashes':{f:hashlib.sha256((v/f).read_bytes()).hexdigest() for f in ['im0.png','im1.png','disp0.pfm','disp1.pfm','calib.txt']}}
        mf['calib_shape']=[int(c['height']),int(c['width'])]
        mf['calib_shape_matches']=mf['shape']==mf['calib_shape']
        manifest.append(mf)
        for scale,configs in [(1.0,FULL),(.5,FULL[:2])]:
            H,W=left.shape;nd=int(np.ceil(c['ndisp']*scale/16)*16);sz=(int(W*scale),int(H*scale))
            l=cv2.resize(left,sz,interpolation=cv2.INTER_AREA) if scale!=1 else left
            r=cv2.resize(right,sz,interpolation=cv2.INTER_AREA) if scale!=1 else right
            ms={name:matcher(kind,b,p2,nd) for name,kind,b,p2 in configs};dur={k:[] for k in ms};outputs={}
            warm=list(ms);rng.shuffle(warm)
            for name in warm:ms[name].compute(l,r)
            for rep in range(3):
                order=list(ms);rng.shuffle(order)
                for rank,name in enumerate(order):
                    start=time.perf_counter();raw=ms[name].compute(l,r);elapsed=(time.perf_counter()-start)*1000
                    if name in outputs:assert np.array_equal(outputs[name],raw),'Nondeterministic output'
                    outputs[name]=raw;dur[name].append(elapsed)
                    record={'view':v.name,'scale':scale,'rep':rep+1,'rank':rank+1,'method':name,'ms':elapsed};times.append(record);schedule.append({k:val for k,val in record.items() if k!='ms'})
            preds={name:(cv2.resize(raw.astype('float32')/16, (W,H),interpolation=cv2.INTER_NEAREST)/scale if scale!=1 else raw.astype('float32')/16) for name,raw in outputs.items()}
            common=np.isfinite(gt)&(gt>=0)&np.logical_and.reduce([d>=0 for name,d in preds.items() if name in ['BM_b9','SGBM_b5_p32']])
            roi=np.zeros_like(gt,dtype=bool);border=int(np.ceil(4/scale));roi[border:H-border,int(nd/scale)+border:W-border]=True;roi&=np.isfinite(gt)&(gt>=0)
            folder=a.out/('full' if scale==1 else 'half')/v.name;folder.mkdir(parents=True,exist_ok=True)
            for name,kind,b,p2 in configs:
                pred=preds[name];res=metrics(pred,gt,c)
                res.update(view=v.name,scene_group=mf['scene_group'],method=name,scale=scale,num_disparities=nd,block=b,p2_coefficient=p2,ms_median=float(np.median(dur[name])),ms_min=min(dur[name]),ms_max=max(dur[name]))
                res['roi_bad2']=100*float(np.mean((pred[roi]<0)|(np.abs(pred[roi]-gt[roi])>2)))
                cm=common&(pred>=0);res['common_epe']=float(np.abs(pred[cm]-gt[cm]).mean());res['common_pixels']=int(cm.sum());rows.append(res)
                np.savez_compressed(folder/(name+'.npz'),disparity=outputs[name].astype('float32')/16,scale=scale)
        print(f'[{i+1}/{len(views)}] {v.name} complete',flush=True)
        write_csv(a.out/'metrics.csv',rows);write_csv(a.out/'timings.csv',times)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');write_csv(a.out/'run_order.csv',schedule)
    aggregate(rows,a.out);meta['finished']=datetime.datetime.now().isoformat();(a.out/'environment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('COMPLETE',a.out,flush=True)

def write_csv(path,rows):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
