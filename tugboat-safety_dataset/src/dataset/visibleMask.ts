import { Color, Material, Mesh, MeshBasicMaterial, NoToneMapping, Object3D, PerspectiveCamera, Scene, ShaderMaterial, SRGBColorSpace, WebGLRenderer, WebGLRenderTarget } from 'three';

/** Shared V1/V2 pass. Preserve vertex deformation, alpha tests and depth;
 * change only fragment RGB, so shader-driven water is never flattened. */
export class VisibleTowlinePass {
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
  capture(gl:WebGLRenderer,scene:Scene,camera:PerspectiveCamera,rope:Mesh,width:number,height:number,exposure=gl.toneMappingExposure) {
    const target=new WebGLRenderTarget(width,height,{samples:0});
    target.texture.colorSpace=SRGBColorSpace;
    const previousTarget=gl.getRenderTarget(),background=scene.background,fog=scene.fog;
    const tone=gl.toneMapping,previousExposure=gl.toneMappingExposure,autoClear=gl.autoClear;
    const clear=gl.getClearColor(new Color()),alpha=gl.getClearAlpha();
    const saved:Array<[Object3D & {material:Material|Material[]},Material|Material[]]>=[];
    const bytes=new Uint8Array(width*height*4);
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
    const context=canvas.getContext('2d');
    if(!context){target.dispose();throw new Error('PNG canvas unavailable');}
    const read=(binary:boolean)=>{
      gl.readRenderTargetPixels(target,0,0,width,height,bytes);
      const pixels=context.createImageData(width,height),mask=new Uint8Array(width*height);
      for(let y=0;y<height;y++)for(let x=0;x<width;x++) {
        const src=((height-1-y)*width+x)*4,dst=(y*width+x)*4;
        const value=bytes[src]>=128?255:0;
        for(let k=0;k<3;k++)pixels.data[dst+k]=binary?value:bytes[src+k];
        pixels.data[dst+3]=255;mask[y*width+x]=value;
      }
      context.putImageData(pixels,0,0);
      return {png:canvas.toDataURL('image/png'),mask};
    };
    try {
      gl.setRenderTarget(target);gl.autoClear=true;gl.toneMappingExposure=exposure;
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
      return {rgb,mask:mask.png,maskPixels:mask.mask};
    } finally {
      for(const [object,material] of saved)object.material=material;
      scene.background=background;scene.fog=fog;gl.toneMapping=tone;gl.toneMappingExposure=previousExposure;gl.autoClear=autoClear;
      gl.setClearColor(clear,alpha);gl.setRenderTarget(previousTarget);target.dispose();
    }
  }
}
