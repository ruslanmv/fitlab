const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const ctx={};vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root,'site/hardware.js'),'utf8'),ctx);
vm.runInContext(fs.readFileSync(path.join(root,'site/hardware-ui.js'),'utf8'),ctx);
const api=ctx.FitLabHardwareUI;
test('desktop, laptop and memory variants remain distinct',()=>{
  assert.equal(api.restore('rtx4060ti-8').vram,8);
  assert.equal(api.restore('rtx4060ti-16').vram,16);
  assert.equal(api.restore('rtx5090-32').vram,32);
  assert.equal(api.restore('rtx5090-laptop-24').vram,24);
  assert.equal(api.restore('missing'),null);
});
test('custom GPU survives URL encoding and rejects invalid data',()=>{
  const g=api.custom('My GPU',20,'');
  assert.equal(g.bw,null);
  const params=new URLSearchParams({gpu:g.id});
  const restored=api.restore(new URLSearchParams(params.toString()).get('gpu'));
  assert.equal(restored.name,g.name);
  assert.equal(restored.vram,20);
  for(const v of [0,-2,NaN,Infinity,300])assert.equal(api.custom('GPU',v,100),null);
  assert.equal(api.custom('<img onerror=alert(1)>',8,200),null);
  assert.equal(api.restore('custom-%not-json'),null);
});
test('both clients use matching standalone assets and parse successfully',()=>{
  for(const asset of ['hardware.js','hardware-ui.js'])assert.equal(
    fs.readFileSync(path.join(root,'site',asset),'utf8'),
    fs.readFileSync(path.join(root,'huggingface-space',asset),'utf8'));
  for(const dir of ['site','huggingface-space']){
    const html=fs.readFileSync(path.join(root,dir,'index.html'),'utf8');
    for(const [,js] of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(js);
    assert.ok(html.includes('src="hardware.js"'));
    assert.ok(!html.includes('const GPUS=['));
  }
});
