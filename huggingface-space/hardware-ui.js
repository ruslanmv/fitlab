/* Shared accessible hardware picker; GPU specs live in generated hardware.js. */
(() => {
  const catalog = globalThis.FitLabHardware;
  const profiles = catalog.gpus.map(g => ({...g, vram:g.vram_gb, bw:g.bandwidth_gbs,
    sub:`${g.vram_gb} GB · ${g.form_factor}`}));
  function custom(name, vram, bw) {
    name = String(name).trim(); vram=Number(vram); bw=bw === '' || bw == null ? null : Number(bw);
    if(!/^[\p{L}\p{N} ._()+-]{1,80}$/u.test(name) || !Number.isFinite(vram) || vram<=0 || vram>256 ||
      (bw!==null && (!Number.isFinite(bw) || bw<=0 || bw>10000))) return null;
    const id='custom-'+encodeURIComponent(JSON.stringify([name,vram,bw]));
    return {id,model:name,name:`${name} · ${vram} GB (custom)`,vram,bw,vram_gb:vram,
      bandwidth_gbs:bw,form_factor:'desktop',vendor:'unknown',arch:'unknown',sm:null,
      notes:'User supplied VRAM. Runtime compatibility is unverified. Bandwidth is optional; speeds are estimates.',sources:[]};
  }
  function restore(id) {
    const known=profiles.find(g=>g.id===id); if(known)return known;
    if(!id?.startsWith('custom-'))return null;
    try {const a=JSON.parse(decodeURIComponent(id.slice(7))); return Array.isArray(a)&&a.length===3?custom(...a):null;}
    catch(e){return null;}
  }
  function mount(select, current, onSelect) {
    const host=select.parentElement;
    const search=document.createElement('input');search.type='search';search.placeholder='Find GPU, e.g. RTX 5060';
    search.setAttribute('aria-label','Search hardware');host.insertBefore(search,select);
    const detail=document.createElement('p');detail.className='mini';detail.setAttribute('aria-live','polite');host.append(detail);
    const more=document.createElement('details'),summary=document.createElement('summary');
    summary.textContent='My GPU is not listed';more.append(summary);
    const fields={};
    for(const [key,label,type] of [['name','GPU name','text'],['vram','VRAM (GB)','number'],['bw','Bandwidth (GB/s, optional)','number']]) {
      const wrap=document.createElement('label');wrap.textContent=label;
      const input=document.createElement('input');input.type=type;
      if(type==='number'){input.min='0.1';input.max=key==='vram'?'256':'10000';input.step='any';}
      else input.maxLength=80;
      wrap.append(input);more.append(wrap);fields[key]=input;
    }
    const button=document.createElement('button');button.type='button';button.textContent='Use custom GPU';more.append(button);
    const error=document.createElement('p');error.setAttribute('role','alert');more.append(error);host.append(more);
    let active=current||profiles.find(g=>g.id==='rtx3060-12');
    function options() {
      select.replaceChildren();const needle=search.value.trim().toLowerCase();
      const groups=new Map();
      for(const g of [...profiles,...(active.id.startsWith('custom-')?[active]:[])]) {
        if(g.id!==active.id && !`${g.name} ${g.arch} ${g.id}`.toLowerCase().includes(needle))continue;
        const groupName=g.form_factor==='laptop'?'Laptop GPUs':g.form_factor==='desktop'?'Desktop GPUs':'Other hardware';
        if(!groups.has(groupName)) {const group=document.createElement('optgroup');group.label=groupName;groups.set(groupName,group);select.append(group);}
        const option=document.createElement('option');option.value=g.id;option.textContent=g.name;groups.get(groupName).append(option);
      }
      select.value=active.id;
    }
    function update(g) {
      active=g;options();
      const age=Math.floor((Date.now()-Date.parse(catalog.updated_at))/864e5);
      detail.textContent=`Catalog ${catalog.updated_at}${age>45?' (over 45 days old)':''} · ${g.arch}`+
        `${g.sm&&g.sm!=='n/a'?' · CUDA CC '+g.sm:''} · ${g.bw?'bandwidth '+g.bw+' GB/s':'bandwidth unknown; no speed estimate'}. `+
        (g.notes||'')+' Selecting hardware changes fit estimates; it does not select a physical GPU.';
    }
    search.addEventListener('input',options);
    select.addEventListener('change',()=>{const g=restore(select.value);if(g){update(g);onSelect(g);}});
    button.onclick=()=>{
      const g=custom(fields.name.value,fields.vram.value,fields.bw.value);
      if(!g){error.textContent='Enter a GPU name, VRAM from 0.1 to 256 GB, and optional positive bandwidth.';return;}
      error.textContent='';search.value='';update(g);onSelect(g);more.open=false;
    };
    update(active);
    return {update};
  }
  globalThis.FitLabHardwareUI={profiles,restore,mount,custom};
})();
