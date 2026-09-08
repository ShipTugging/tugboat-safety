import { Color, Material, Mesh, MeshBasicMaterial, NoToneMapping, Object3D, PerspectiveCamera, Scene, ShaderMaterial, SRGBColorSpace, WebGLRenderer, WebGLRenderTarget, type TubeGeometry } from 'three';
import { makeMetadata, type TowlineCapture } from './geometry';
import type { CaptureSample } from './types';

// Retain each occluder's original vertex deformation, alpha tests and depth state.
// Only its final RGB output changes. In particular, the ocean's wave surface
// remains identical in RGB and mask, unlike a blanket scene.overrideMaterial.
export class TowlineMaskPass {
  private cache=new Map<Material,Material>();
  private white=new MeshBasicMaterial({color:0xffffff,toneMapped:false,fog:false});
  private black(original:Material) {
    let material=this.cache.get(original);
    if(!material) {
      material=original.clone();
      const blacken=(source:string)=>{
        const end=source.lastIndexOf('}');
        return source.slice(0,end)+'\ngl_FragColor.rgb=vec3(0.0);\n'+source.slice(end);
      };
      if(material instanceof ShaderMaterial && original instanceof ShaderMaterial) {
        material.uniforms=original.uniforms;
        material.fragmentShader=blacken(material.fragmentShader);
      } else {
        material.onBeforeCompile=shader=>{shader.fragmentShader=blacken(shader.fragmentShader);};
        material.customProgramCacheKey=()=>`towline-black-${original.type}`;
      }
      this.cache.set(original,material);
    }
    return material;
  }
  dispose() {this.white.dispose();for(const m of this.cache.values())m.dispose();this.cache.clear();}
  capture(gl:WebGLRenderer,scene:Scene,camera:PerspectiveCamera,rope:Mesh,sample:CaptureSample):TowlineCapture {
    const target=new WebGLRenderTarget(sample.width,sample.height,{samples:0});
    target.texture.colorSpace=SRGBColorSpace;
    const previousTarget=gl.getRenderTarget(),background=scene.background,fog=scene.fog;
    const tone=gl.toneMapping,exposure=gl.toneMappingExposure,autoClear=gl.autoClear;
    const clear=gl.getClearColor(new Color()),alpha=gl.getClearAlpha();
    const saved:Array<[Object3D & {material:Material|Material[]},Material|Material[]]>=[];
    const bytes=new Uint8Array(sample.width*sample.height*4);
    const canvas=document.createElement('canvas');canvas.width=sample.width;canvas.height=sample.height;
    const context=canvas.getContext('2d')!;
    const read=(binary:boolean)=>{
      gl.readRenderTargetPixels(target,0,0,sample.width,sample.height,bytes);
      const pixels=context.createImageData(sample.width,sample.height),mask=new Uint8Array(sample.width*sample.height);
      for(let y=0;y<sample.height;y++)for(let x=0;x<sample.width;x++) {
        const src=((sample.height-1-y)*sample.width+x)*4,dst=(y*sample.width+x)*4;
        const value=bytes[src]>=128?255:0;
        for(let k=0;k<3;k++)pixels.data[dst+k]=binary?value:bytes[src+k];
        pixels.data[dst+3]=255;mask[y*sample.width+x]=value;
      }
      context.putImageData(pixels,0,0);
      return {png:canvas.toDataURL('image/png'),mask};
    };
    try {
      gl.setRenderTarget(target);gl.autoClear=true;gl.toneMappingExposure=sample.towline!.exposure;
      gl.render(scene,camera);const rgb=read(false).png;
      scene.traverse(object=>{
        if('material' in object) {
          const renderable=object as Object3D & {material:Material|Material[]};
          const original=renderable.material;saved.push([renderable,original]);
          renderable.material=object===rope?this.white:Array.isArray(original)?original.map(m=>this.black(m)):this.black(original);
        }
      });
      scene.background=new Color(0);scene.fog=null;gl.toneMapping=NoToneMapping;gl.setClearColor(0,1);
      gl.render(scene,camera);const mask=read(true);
      const curve=(rope.geometry as TubeGeometry).parameters.path;
      // Geometry lives at world origin today; apply matrixWorld for future transforms.
      const worldCurve=Object.create(curve) as typeof curve;
      worldCurve.getSpacedPoints=(divisions:number)=>curve.getSpacedPoints(divisions).map(p=>p.applyMatrix4(rope.matrixWorld));
      return {rgb,mask:mask.png,metadata:makeMetadata(sample,camera,worldCurve,mask.mask)};
    } finally {
      for(const [object,material] of saved)object.material=material;
      scene.background=background;scene.fog=fog;gl.toneMapping=tone;gl.toneMappingExposure=exposure;gl.autoClear=autoClear;
      gl.setClearColor(clear,alpha);gl.setRenderTarget(previousTarget);target.dispose();
    }
  }
}
