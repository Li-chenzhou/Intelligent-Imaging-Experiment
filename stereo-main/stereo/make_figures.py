from pathlib import Path
import sys,os,json,csv
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor'))
os.environ['MPLCONFIGDIR']=str(BASE/'qa/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PIL import Image
from stereo_experiment import read_pfm,calib,np

def main():
    root=BASE.parent/'all/data';out=BASE/'results';figdir=out/'figures';figdir.mkdir(exist_ok=True)
    result=json.loads((out/'summary.json').read_text());s=result['summary']
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'figure.facecolor':'white'}):
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():continue
            gt=read_pfm(folder/'disp0.pfm');c=calib(folder/'calib.txt');valid=np.isfinite(gt)&(gt>=0)
            bm=np.load(out/'full'/folder.name/'BM_b9.npz')['disparity'];sg=np.load(out/'full'/folder.name/'SGBM_b5_p32.npz')['disparity']
            cm=plt.get_cmap('viridis').copy();cm.set_bad('#bbbbbb')
            fig,axs=plt.subplots(3,2,figsize=(9,9))
            fig.subplots_adjust(left=.015,right=.985,top=.94,bottom=.17,hspace=.22,wspace=.03)
            for ax,im,title in [(axs[0,0],Image.open(folder/'im0.png'),'Left image'),(axs[0,1],Image.open(folder/'im1.png'),'Right image')]:
                ax.imshow(im);ax.set_title(title)
            for ax,arr,title in [(axs[1,0],gt,'Ground truth'),(axs[1,1],bm,'BM block 9'),(axs[2,0],sg,'SGBM block 5 / P2=32b²')]:
                m=np.isfinite(arr)&(arr>=0);ax.imshow(np.ma.masked_where(~m,arr),cmap=cm,vmin=0,vmax=c['ndisp']);ax.set_title(title)
            diff=np.full(gt.shape,np.nan,dtype=np.float32);mask=valid&(sg>=0);diff[mask]=np.abs(sg[mask]-gt[mask])
            ec=plt.get_cmap('magma').copy();ec.set_bad('#bbbbbb');colors=ec(Normalize(0,10,clip=True)(np.nan_to_num(diff)))
            colors[~valid]=[.73,.73,.73,1];colors[valid&(sg<0)]=[1,0,1,1]
            axs[2,1].imshow(colors);axs[2,1].set_title('SGBM absolute error (clipped at 10 px)')
            for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
            dcax=fig.add_axes([.07,.11,.36,.018]);ecax=fig.add_axes([.57,.11,.36,.018])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,c['ndisp']),cmap=cm),cax=dcax,orientation='horizontal',label='Disparity (original px)')
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,10),cmap=ec),cax=ecax,orientation='horizontal',label='Absolute error (original px)')
            fig.legend(handles=[Patch(color='magenta',label='Error map: invalid prediction'),Patch(color='#bbbbbb',label='Error map: invalid GT')],loc='lower center',ncol=2,fontsize=9,bbox_to_anchor=(.5,.005))
            fig.suptitle(folder.name,fontsize=13);fig.savefig(figdir/f'{folder.name}.png',dpi=200);plt.close(fig)
        groups=sorted(s['1.0_BM_b9']['bygroup'])
        fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
        for name,col,marker,label in [('BM_b9','#0072B2','o','BM b9'),('SGBM_b5_p32','#D55E00','s','SGBM b5 p32')]:
            val=[s['1.0_'+name]['bygroup'][g]['bad2_total'] for g in groups]
            ax.plot(groups,val,marker=marker,color=col,label=label,lw=1)
        ax.set_ylabel('Bad2-total (%)');ax.set_ylim(0,100);ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2);ax.legend();fig.savefig(figdir/'scene_comparison.png',dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,3.6),layout='constrained')
        for coeff,col,mark in [(32,'#0072B2','o'),(128,'#D55E00','s')]:
            y=[s[f'1.0_SGBM_b{b}_p{coeff}']['scene_macro']['bad2_total'] for b in [5,9]]
            ax.plot([5,9],y,marker=mark,label=f'P2={coeff}b²',color=col)
        ax.set(xlabel='Block size (pixels)',ylabel='Scene-macro Bad2-total (%)',xticks=[5,9]);ax.grid(alpha=.2);ax.legend();fig.savefig(figdir/'factorial.png',dpi=220);plt.close(fig)
    print('Figures written',len(list(figdir.glob('*.png'))))
if __name__=='__main__':main()
