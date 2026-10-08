/* Run with Playwright available via NODE_PATH; all registry reads are local fixtures. */
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const server=spawn(process.env.PYTHON||'python',['-m','http.server','8879','--bind','127.0.0.1'],{cwd:root,stdio:'ignore'});
(async()=>{
 let browser;
 try{
  for(let i=0;i<100;i++){
   try{await fetch('http://127.0.0.1:8879/site/index.html');break;}catch(e){await new Promise(r=>setTimeout(r,100));}
  }
  browser=await chromium.launch({headless:true,args:['--no-sandbox']});
  for(const dir of ['site','huggingface-space']){
   const page=await browser.newPage({viewport:{width:1280,height:900}}), errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.route('https://raw.githubusercontent.com/**',route=>route.fulfill({
     contentType:'application/json',body:fs.readFileSync(path.join(root,'site/registry.json'),'utf8')}));
   await page.goto(`http://127.0.0.1:8879/${dir}/index.html?gpu=rtx4060ti-8`);
   const selector=dir==='site'?'#gpuSelect':'#gpu';
   await page.locator(selector).waitFor();
   assert.equal(await page.locator(selector).inputValue(),'rtx4060ti-8');
   assert.ok(await page.locator(selector+' option').count()>=85);
   await page.locator(selector).selectOption('rtx4060ti-16');
   const actual=await page.evaluate(isSite=>isSite?gpu.vram:gpuNow().vram,dir==='site');
   assert.equal(actual,16);
   await page.getByRole('searchbox',{name:'Search hardware'}).fill('5090');
   const options=await page.locator(selector+' option').count();assert.ok(options<=4);
   await page.locator(selector).selectOption('rtx5090-laptop-24');
   assert.equal(await page.evaluate(isSite=>isSite?gpu.vram:gpuNow().vram,dir==='site'),24);
   await page.getByText('My GPU is not listed',{exact:true}).click();
   const more=page.locator(selector).locator('..').locator('details');
   await more.locator('input[type=text]').fill('Future GPU');
   await more.locator('input[type=number]').first().fill('20');
   await page.getByRole('button',{name:'Use custom GPU',exact:true}).click();
   assert.equal(await page.evaluate(isSite=>isSite?gpu.bw:gpuNow().bw,dir==='site'),null);
   assert.ok((await page.locator(selector).inputValue()).startsWith('custom-'));
   const url=page.url();await page.reload();await page.locator(selector).waitFor();
   assert.equal(page.url(),url);
   assert.equal(await page.evaluate(isSite=>isSite?gpu.vram:gpuNow().vram,dir==='site'),20);
   assert.ok((await page.locator('body').innerText()).includes('bandwidth unknown'));
   await page.setViewportSize({width:390,height:844});
   assert.ok(await page.locator(selector).isVisible());
   assert.deepEqual(errors,[]);
   if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,dir+'.png'),fullPage:true});
   console.log(dir+': GPU variants, search, custom VRAM, URL restore, mobile and runtime checks passed');
   await page.close();
  }
 }finally{if(browser)await browser.close();server.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
