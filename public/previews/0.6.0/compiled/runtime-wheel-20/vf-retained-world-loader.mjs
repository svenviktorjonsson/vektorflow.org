// Retained-world startup orchestration only. The supplied compiled-law adapter
// owns GPU state; an inactive World must never block the selected View.
export function createRetainedWorldLoader(worlds, initialize) {
  const applications=[], pending=new Map();
  let disposed=false;
  return {
    applications,
    async ensure(worldId) {
      if(disposed)throw Error('World loader is disposed');
      const existing=applications.find(app=>app.world.world_id===worldId);
      if(existing)return existing;
      const world=worlds.find(world=>world.world_id===worldId);
      if(!world)throw Error('Selected View has no compiled World');
      if(!pending.has(worldId))pending.set(worldId,Promise.resolve().then(()=>initialize(world)).then(app=>{
        if(disposed){app.destroy();throw Error('World loader is disposed');}
        applications.push(app);return app;
      }).finally(()=>pending.delete(worldId)));
      return pending.get(worldId);
    },
    destroy(){disposed=true;for(const app of applications)app.destroy();applications.length=0;},
  };
}
