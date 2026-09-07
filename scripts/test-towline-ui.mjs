import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import JSZip from 'jszip';

const server=await createServer({logLevel:'error',server:{host:'127.0.0.1',port:5193,strictPort:true}});
await server.listen();
const browser=await chromium.launch({channel:'chrome',headless:true});
try {
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(()=>window.towlineDataset?.ready,null,{timeout:60000});
  const section=page.getByRole('region',{name:'Towline dataset generator'});
  await section.getByRole('switch').check();
  await section.getByRole('button',{name:'Preview Random Scenario'}).click();
  await page.getByText('미리보기 완료',{exact:true}).waitFor();
  const first=await section.locator('pre').textContent();
  const data=JSON.parse(first);assert.equal(data.towline.centerlineWorld.length,50);
  await section.getByRole('button',{name:'Preview Random Scenario'}).click();
  await page.waitForFunction(seed=>JSON.parse(document.querySelector('.dataset-preview pre').textContent).scenarioSeed!==seed,data.scenarioSeed);
  await page.getByText('미리보기 완료',{exact:true}).waitFor();
  await section.getByRole('spinbutton',{name:'Image Width'}).fill('640');
  await section.getByRole('spinbutton',{name:'Image Height'}).fill('360');
  const downloading=page.waitForEvent('download');
  await section.getByRole('button',{name:'Generate Single Sample'}).click();
  const download=await downloading;
  await mkdir('artifacts/ui',{recursive:true});
  await download.saveAs('artifacts/ui/towline_dataset.zip');
  const zip=await JSZip.loadAsync(await readFile('artifacts/ui/towline_dataset.zip'));
  const metadata=JSON.parse(await zip.file(/metadata\/train\/.*json$/)[0].async('string'));
  assert.equal(metadata.image.width,640);assert.equal(metadata.image.height,360);
  assert.equal(zip.file(/\.png$/).length,2);
  assert.equal(JSON.parse(await zip.file('towline_dataset/generation_manifest.json').async('string')).status,'complete');
  // Invalid config must be visible to the operator, not swallowed by the event handler.
  await section.getByRole('spinbutton',{name:'Dataset Size'}).fill('0');
  await section.getByRole('button',{name:'Generate Pilot Dataset'}).click();
  await section.getByRole('alert').filter({hasText:'Invalid datasetSize'}).waitFor({timeout:5000});
  // Exercise the folder writer boundary without an OS permission dialog.
  await page.evaluate(()=>{
    window.testFiles={};
    function directory(prefix='') {return {
      async *values(){},
      getDirectoryHandle:async name=>directory(prefix+name+'/'),
      getFileHandle:async name=>({createWritable:async()=>({write:async data=>{window.testFiles[prefix+name]=typeof data==='string'?data:await data.text();},close:async()=>{}})})
    };}
    window.showDirectoryPicker=async()=>directory();
  });
  await section.getByRole('spinbutton',{name:'Dataset Size'}).fill('11');
  await section.getByRole('button',{name:'Generate Pilot Dataset'}).click();
  await page.getByText('11장 저장 완료 · 제외 0장',{exact:true}).waitFor({timeout:60000});
  const saved=await page.evaluate(()=>Object.keys(window.testFiles));
  assert.equal(saved.filter(s=>s.includes('/metadata/')).length,11);
  // Cancel after the folder is selected. Completed samples are retained, status incomplete.
  await section.getByRole('spinbutton',{name:'Dataset Size'}).fill('500');
  await section.getByRole('button',{name:'Generate Pilot Dataset'}).click();
  await section.getByRole('button',{name:'생성 취소'}).click();
  await page.getByText('생성 취소됨',{exact:true}).waitFor();
  assert.equal(await section.getByRole('spinbutton',{name:'Dataset Size'}).isEnabled(),true);
  const cancelled=await page.evaluate(()=>JSON.parse(window.testFiles['towline_dataset/generation_manifest.json']));
  assert.equal(cancelled.status,'incomplete');
  await section.getByRole('button',{name:'Preview Random Scenario'}).click();
  await page.getByText('미리보기 완료',{exact:true}).waitFor();
  await section.screenshot({path:resolve('artifacts/ui/controls.png')});
  // Analytic render oracle: magenta rope, black occluder. Every mask pixel must
  // equal visible magenta RGB support on the identical raster grid.
  const maskTest=await page.evaluate(async()=>{
    const T=await import('/node_modules/three/build/three.module.js');
    const {TowlineMaskPass}=await import('/src/dataset/mask.ts');
    const {makeScenario,planScenarios}=await import('/src/dataset/scenarios.ts');
    const {datasetConfig}=await import('/src/dataset/config.ts');
    const config={...datasetConfig,datasetSize:1,imageWidth:128,imageHeight:128};
    const sample=makeScenario(config,0,0,planScenarios(config)[0]);
    const renderer=new T.WebGLRenderer({antialias:false});renderer.setSize(128,128);renderer.toneMapping=T.NoToneMapping;
    const scene=new T.Scene();scene.background=new T.Color(0);
    const camera=new T.PerspectiveCamera(50,1,.1,100);camera.position.set(0,0,10);camera.lookAt(0,0,0);camera.updateMatrixWorld(true);
    const curve=new T.CatmullRomCurve3([new T.Vector3(-3,0,0),new T.Vector3(0,-1,0),new T.Vector3(3,0,0)]);
    const rope=new T.Mesh(new T.TubeGeometry(curve,128,.12,8,false),new T.MeshBasicMaterial({color:0xff00ff}));
    const box=new T.Mesh(new T.BoxGeometry(1.5,4,1),new T.MeshBasicMaterial({color:0}));box.position.z=1;
    scene.add(rope,box);scene.updateMatrixWorld(true);
    const pass=new TowlineMaskPass(),original=rope.material;
    const frame=pass.capture(renderer,scene,camera,rope,sample);
    const pixels=async url=>{const image=new Image();image.src=url;await image.decode();const c=document.createElement('canvas');c.width=c.height=128;const ctx=c.getContext('2d');ctx.drawImage(image,0,0);return ctx.getImageData(0,0,128,128).data;};
    const rgb=await pixels(frame.rgb),mask=await pixels(frame.mask);let mismatch=0,white=0;
    for(let i=0;i<rgb.length;i+=4){if((rgb[i]>128&&rgb[i+2]>128)!==(mask[i]===255))mismatch++;if(mask[i]===255)white++;}
    const stateRestored=rope.material===original&&renderer.getRenderTarget()===null&&renderer.toneMapping===T.NoToneMapping;
    pass.dispose();renderer.dispose();rope.geometry.dispose();rope.material.dispose();box.geometry.dispose();box.material.dispose();
    return {mismatch,white,stateRestored};
  });
  assert.equal(maskTest.mismatch,0);assert.ok(maskTest.white>0);assert.ok(maskTest.stateRestored);
  const rejected=await page.evaluate(async()=>{
    window.__towlineWrite=async(path,data)=>{window.testFiles[path]=data;};
    const {datasetConfig}=await import('/src/dataset/config.ts');
    try {
      await window.towlineDataset.generate({datasetSize:1,imageWidth:128,imageHeight:128,validation:{...datasetConfig.validation,minMaskPixels:10000000,maxAttemptsPerSample:2}});
      throw new Error('Expected rejection exhaustion');
    } catch(error) {
      const manifest=JSON.parse(window.testFiles['generation_manifest.json']);
      const log=JSON.parse(window.testFiles['rejection_log.json']);
      return {message:error.message,manifest,log};
    }
  });
  assert.match(rejected.message,/exceeded 2 attempts/);
  assert.equal(rejected.manifest.status,'incomplete');assert.equal(rejected.manifest.generatedSamples,0);assert.equal(rejected.manifest.rejectedSamples,2);
  assert.notEqual(rejected.log[0].scenarioSeed,rejected.log[1].scenarioSeed);
  assert.ok(rejected.log.every(entry=>entry.reasons.includes('empty or tiny mask')));
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({preview:true,singleZip:true,customResolution:true,invalidConfigFeedback:true,folderWriter:'mocked browser directory API; 11 samples',cancelAndResume:true,retryExhaustion:true,maskPixelOracle:maskTest},null,2));
} finally {await browser.close();await server.close();}
