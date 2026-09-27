// Adapter policy only. GPU kernels own collision truth; this controller owns
// timing observations and submission choices, never particle state.
export class ContactSchedulerPolicy {
  constructor(){this.reset();}
  reset(){this.mode='parallel';this.frames=0;this.dwell=0;this.lastTime=0;this.stalled=0;this.profiles=new Map();this.forced=null;}
  force(mode){if(!['auto','parallel','ledger'].includes(mode))throw new Error('Unknown contact scheduler');this.forced=mode==='auto'?null:mode;}
  choose(moving=false){
    if(this.forced)return this.forced;
    if(this.stalled>0&&this.frames%64!==0)return 'parallel';
    const p=this.profiles.get(moving?'moving':'stationary');
    if(!p||p.parallel.n<3||p.ledger.n<3)return this.frames%2?'ledger':'parallel';
    if(this.dwell<4)return this.mode;
    // High invalidation defeats retained predictions. Periodic bounded probes
    // allow recovery when the workload changes; no future timing oracle.
    if(this.frames%64===0)return this.mode==='parallel'?'ledger':'parallel';
    if(p.dirty>.35||p.rebuilds>.5)return 'parallel';
    const current=p[this.mode],other=p[this.mode==='parallel'?'ledger':'parallel'];
    const uncertainty=2*Math.sqrt(current.variance/current.n+other.variance/other.n);
    // Amortize a conservative cold-cache surcharge over the minimum dwell.
    const switchCost=this.mode==='parallel'?p.cold/4:0;
    if(other.mean+uncertainty+switchCost<current.mean*.85)return this.mode==='parallel'?'ledger':'parallel';
    return this.mode;
  }
  observe({mode,moving,gpuMs,completedMs,time,events,dirtyParticles,heapRebuilds,fallbacks}){
    this.frames++;this.dwell=mode===this.mode?this.dwell+1:1;this.mode=mode;
    const progress=time-this.lastTime;this.lastTime=time;
    if(!(progress>1e-9)||!(events>0)){this.stalled++;return;}this.stalled=0;
    const key=moving?'moving':'stationary';let p=this.profiles.get(key);
    if(!p){p={parallel:{n:0,mean:0,variance:0},ledger:{n:0,mean:0,variance:0},dirty:0,rebuilds:0,cold:0};this.profiles.set(key,p);}
    const cost=(completedMs??gpuMs)/progress,s=p[mode],delta=cost-s.mean;
    if(!s.n){s.mean=cost;}else{s.mean+=.2*delta;s.variance=.8*(s.variance+.2*delta*delta);}s.n=Math.min(16,s.n+1);
    p.dirty=.8*p.dirty+.2*dirtyParticles;p.rebuilds=.8*p.rebuilds+.2*heapRebuilds/events;
    if(mode==='ledger'&&heapRebuilds)p.cold=.8*p.cold+.2*cost;
    if(fallbacks){p.rebuilds=1;this.mode='parallel';this.dwell=4;}
  }
}
