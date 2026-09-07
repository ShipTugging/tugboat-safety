import { chromium } from 'playwright';
import { createServer } from 'vite';
import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { resolve, dirname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { datasetConfig, validateConfig } from '../src/dataset/config.ts';

const args=process.argv.slice(2),options={};
const allowed=new Set(['count','seed','width','height','out','config','browser']);
for(let i=0;i<args.length;i+=2) {
  const key=args[i].replace(/^--/,'');
  if(!args[i].startsWith('--')||!allowed.has(key)||args[i+1]===undefined)throw new Error('Usage: npm run dataset:generate -- [--count 500] [--seed 1043] [--width 1280] [--height 720] [--out towline_dataset] [--config config.json] [--browser chrome|chromium]');
  options[key]=args[i+1];
}
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const config=structuredClone(datasetConfig);
if(options.config) {
  const overrides=JSON.parse(await readFile(resolve(options.config),'utf8'));
  for(const [key,value] of Object.entries(overrides)) {
    if(!(key in config))throw new Error(`Unknown config key: ${key}`);
    config[key]=value && typeof value==='object'&&!Array.isArray(value)?{...config[key],...value}:value;
  }
}
for(const [arg,key] of [['count','datasetSize'],['seed','seed'],['width','imageWidth'],['height','imageHeight']])if(options[arg]!==undefined)config[key]=Number(options[arg]);
validateConfig(config);
if(options.browser&&!['chrome','chromium'].includes(options.browser))throw new Error('Browser must be chrome or chromium');
const output=resolve(options.out??'towline_dataset');
await mkdir(dirname(output),{recursive:true});
await mkdir(output); // Refuse overwriting an existing run, even if it is incomplete.
let browser,server;
try {
  server=await createServer({root,logLevel:'error',server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({channel:options.browser==='chromium'?undefined:'chrome',headless:true,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on('pageerror',error=>console.error('Browser error:',error.message));
  let count=0;
  await page.exposeFunction('__towlineWrite',async(path,data,encoding)=>{
    const file=resolve(output,path);
    if(!file.startsWith(output+sep)||!['utf8','base64'].includes(encoding))throw new Error('Invalid export path or encoding');
    await mkdir(dirname(file),{recursive:true});
    await writeFile(file+'.partial',Buffer.from(data,encoding));await rename(file+'.partial',file);
    if(path.startsWith('metadata/')) {count++;if(count===1||count%10===0||count===config.datasetSize)console.log(`Generating: ${count} / ${config.datasetSize}`);}
  });
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(()=>window.towlineDataset?.ready,{},{timeout:60000});
  const result=await page.evaluate(config=>window.towlineDataset.generate(config),config);
  await writeFile(resolve(output,'runtime.json'),JSON.stringify({node:process.version,browser:browser.version(),platform:process.platform,sourceCommit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),workingTreeModified:!!execFileSync('git',['status','--porcelain'],{cwd:root,encoding:'utf8'}).trim()},null,2)+'\n');
  console.log(JSON.stringify({output,...result},null,2));
} finally {await browser?.close();await server?.close();}
