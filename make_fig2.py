from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
W=pd.read_csv(DATA/'mean_waveforms.csv')
T=pd.read_csv(DATA/'trials.csv')
S=pd.read_csv(DATA/'relative_spikes.csv')
Q=pd.read_csv(DATA/'quality_metrics.csv')
BLUE='#005B8E';ORANGE='#B44800';GREY='#49535C'
EDGES=np.arange(-.5,1.000001,.1)
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.titlesize':7,'axes.labelsize':7,
                     'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
                     'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
                     'xtick.major.size':2.5,'ytick.major.size':2.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',pad=2)
OUT=HERE/'figures';OUT.mkdir(parents=True,exist_ok=True)
SPECS=[('ZYH_G003','Participant 1'),('ZZJ008_S2_6_9','Participant 2')]

def read_phase_spikes(name):
    d=pd.read_csv(HERE/'data/phase_spikes.csv')
    d=d[d.subject==name]
    return {phase:[g.spike_time_s.to_numpy() for _,g in d[d.phase==phase].groupby('segment',sort=True)]
            for phase in ('Learning','Testing')}

def acg_lags(segments,maxlag=.05):
    out=[]
    for t in segments:
        for i,x in enumerate(t):
            j=i+1
            while j<len(t) and t[j]-x<=maxlag:
                out.extend((1000*(t[j]-x),-1000*(t[j]-x)))
                j+=1
    return np.asarray(out)

def intervals(segments):
    return np.concatenate([np.diff(t)*1000 for t in segments if len(t)>1])

def wave(ax,name):
    data=W[W.subject==name]
    offset=55 if name=='ZYH_G003' else 100
    for phase,color in [('Learning',BLUE),('Testing',ORANGE)]:
        for wire in range(1,5):
            d=data[(data.phase==phase)&(data.wire==wire)]
            ax.plot(d.time_ms,d.mean_uv+offset*(wire-1),lw=1.35,color=color)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim((-65,215) if name=='ZYH_G003' else (-175,400))
    ax.set_xlabel('Time from spike (ms)')
    ax.set_ylabel(f'µV + {offset} µV/contact')
    phase_cos=float(Q[Q.subject==name].pooled_waveform_cosine.iloc[0])
    ax.text(.98,.96,f'Phase-mean cosine {phase_cos:.4f}',transform=ax.transAxes,
            ha='right',va='top',fontsize=5.8)
    clean(ax)

def waveform_consistency(ax,name):
    d=pd.read_csv(HERE/'data/single_spike_cosine.csv')
    d=d[d.subject==name]
    for i,(phase,color) in enumerate([('Learning',BLUE),('Testing',ORANGE)]):
        v=d[d.phase==phase].cosine_to_pooled_mean.to_numpy()
        ax.boxplot([v],positions=[i],widths=.48,patch_artist=True,showfliers=False,
                   boxprops=dict(facecolor=color,alpha=.35,edgecolor=color),
                   medianprops=dict(color=color,lw=1.3),whiskerprops=dict(color=color),capprops=dict(color=color))
        ax.scatter(i+np.random.default_rng(31+i).uniform(-.16,.16,len(v)),v,s=3,alpha=.18,color=color,rasterized=True)
    ax.set(xlim=(-.5,1.5),ylim=(0,1.02),xticks=[0,1],xticklabels=['Learning','Testing'],ylabel='Single-spike cosine')
    clean(ax)

def lag_hist(ax,phases,phase,kind):
    color=BLUE if phase=='Learning' else ORANGE
    if kind=='acg':
        edges=np.arange(-51,52,2);x=acg_lags(phases[phase]);lim=(-50,50)
        xlabel='Spike lag (ms)';shade=(-2,2)
    else:
        edges=np.arange(0,21,1);x=intervals(phases[phase]);lim=(0,20)
        xlabel='Inter-spike interval (ms)';shade=(0,2)
    hist,_=np.histogram(x,edges)
    ax.bar(edges[:-1],hist,width=np.diff(edges),align='edge',color=color,edgecolor='white',linewidth=.25)
    ax.axvspan(*shade,color='#d9dcdf',alpha=.5,zorder=0)
    ax.set(xlim=lim)
    if kind=='isi':
        ax.axvline(2,color=GREY,lw=.75,ls='--')
        ax.text(.98,.93,f'{phase}: <2 ms {np.count_nonzero(x<2)}/{len(x)}',transform=ax.transAxes,ha='right',va='top',fontsize=5.5)
    else:
        ax.text(.98,.93,f'{phase}  n={sum(len(t) for t in phases[phase])}',transform=ax.transAxes,ha='right',va='top',fontsize=6)
    if phase=='Testing':ax.set_xlabel(xlabel)
    else:ax.tick_params(labelbottom=False)
    clean(ax)

def raster(ax,name):
    offset=0
    for phase,color in [('Learning',BLUE),('Testing',ORANGE)]:
        tr=T[(T.subject==name)&(T.phase==phase)&T.analysis_included].sort_values(['session','trial'])
        sp=S[(S.subject==name)&(S.phase==phase)]
        arr=[sp[(sp.session==r.session)&(sp.trial==r.trial)].time_from_onset_s.to_numpy() for _,r in tr.iterrows()]
        ax.eventplot(arr,lineoffsets=np.arange(len(arr))+offset+.5,linelengths=2.2,linewidths=1.15,colors=color)
        offset+=len(arr)
        if phase=='Learning':ax.axhline(offset,color=GREY,lw=.8,ls='--')
    ax.axvline(0,color=GREY,lw=.9)
    ax.set(xlim=(-.5,1),ylim=(0,offset),ylabel='Trial')
    ax.set_xticks([-.5,0,.5,1]);ax.tick_params(labelbottom=False)
    clean(ax)

def psth(ax,name):
    width=.045
    for phase,color,shift in [('Learning',BLUE,-width/2),('Testing',ORANGE,width/2)]:
        tr=T[(T.subject==name)&(T.phase==phase)&T.analysis_included]
        sp=S[(S.subject==name)&(S.phase==phase)]
        counts,_=np.histogram(sp.time_from_onset_s,EDGES)
        rate=counts/(len(tr)*.1)
        ax.bar(EDGES[:-1]+.05+shift,rate,width=width,color=color,label=phase,edgecolor='none')
    ax.axvline(0,color=GREY,lw=.8)
    ax.set(xlim=(-.5,1),ylim=(0,3),ylabel='Rate (Hz)',xlabel='Time from picture onset (s)')
    ax.set_xticks([-.5,0,.5,1]);ax.set_yticks([0,1,2,3]);clean(ax)
    ax.set_ylim(0,1.7 if name=='ZYH_G003' else 1.1)

def label(ax,s):ax.text(-.16,1.07,s,transform=ax.transAxes,fontweight='bold',fontsize=9,clip_on=False)

def main():
    fig=plt.figure(figsize=(7.3,6.45))
    grid=fig.add_gridspec(3,4,left=.085,right=.98,top=.82,bottom=.09,hspace=.64,wspace=.5,
                          height_ratios=[1.0,1.0,2.0])
    fig.suptitle('Unit quality and picture-onset responses',x=.085,ha='left',fontsize=11,fontweight='bold',y=.98)
    handles=[Line2D([0],[0],color=BLUE,lw=2,label='Learning'),Line2D([0],[0],color=ORANGE,lw=2,label='Testing'),
             Patch(facecolor='#d9dcdf',label='Refractory interval: <2 ms')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.085,.953),ncol=3,frameon=False,
               fontsize=7,columnspacing=1.5,handlelength=1.8)
    for k,(name,title) in enumerate(SPECS):
        c=2*k;fig.text(.29 if k==0 else .76,.885,title,ha='center',fontsize=9,fontweight='bold')
        phases=read_phase_spikes(name)
        for j,heading in enumerate(['Waveform','Waveform consistency']):
            ax=fig.add_subplot(grid[0,c+j])
            if j==0:wave(ax,name)
            else:waveform_consistency(ax,name)
            ax.set_title(heading,loc='left',fontsize=7,pad=4)
            label(ax,chr(ord('a')+2*j+k))
        for j,(kind,heading) in enumerate([('acg','Autocorrelogram (pairs / 2 ms)'),('isi','Short ISIs (1-ms bins)')]):
            sub=grid[1,c+j].subgridspec(2,1,hspace=.12)
            axs=[fig.add_subplot(sub[0]),fig.add_subplot(sub[1])]
            for ax,phase in zip(axs,['Learning','Testing']):lag_hist(ax,phases,phase,kind)
            axs[0].set_title(heading,loc='left',fontsize=7,pad=4)
            label(axs[0],chr(ord('e')+2*j+k))
        task=grid[2,c:c+2].subgridspec(2,1,height_ratios=[1.7,.8],hspace=.18)
        r=fig.add_subplot(task[0]);raster(r,name);r.set_title('Picture-onset raster',loc='left',fontsize=8,pad=4);label(r,'i' if k==0 else 'j')
        n_l=len(T[(T.subject==name)&(T.phase=='Learning')&T.analysis_included]);n_t=len(T[(T.subject==name)&(T.phase=='Testing')&T.analysis_included])
        r.text(.98,1.04,f'Learning {n_l}  ·  Testing {n_t}',transform=r.transAxes,ha='right',va='bottom',fontsize=7)
        p=fig.add_subplot(task[1],sharex=r);psth(p,name)
        p.text(.02,.96,'PSTH · 100 ms bins',transform=p.transAxes,ha='left',va='top',fontsize=7,
               bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.5))
        p.text(-.16,.95,'k' if k==0 else 'l',transform=p.transAxes,fontsize=9,fontweight='bold')
    fig.text(.085,.025,'Trial window: −0.5 to +1.0 s; pre-onset activity is context. Spike rasters and PSTHs use the same included trials.',fontsize=6.5)
    fig.savefig(OUT/'Fig2_reproduced.png',dpi=400);fig.savefig(OUT/'Fig2_reproduced.pdf');plt.close(fig)

if __name__=='__main__':main()
