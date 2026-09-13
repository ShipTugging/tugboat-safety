import { Mesh, PerspectiveCamera, Scene, WebGLRenderer, type TubeGeometry } from 'three';
import { makeMetadata, type TowlineCapture } from './geometry';
import type { CaptureSample } from './types';
import { VisibleTowlinePass } from './visibleMask';

/** V1 API and metadata stay intact; V2 reuses the same pixel renderer. */
export class TowlineMaskPass {
  private pass=new VisibleTowlinePass();
  dispose() {this.pass.dispose();}
  capture(gl:WebGLRenderer,scene:Scene,camera:PerspectiveCamera,rope:Mesh,sample:CaptureSample):TowlineCapture {
    const pixels=this.pass.capture(gl,scene,camera,rope,sample.width,sample.height,sample.towline!.exposure);
    const curve=(rope.geometry as TubeGeometry).parameters.path;
    const worldCurve=Object.create(curve) as typeof curve;
    worldCurve.getSpacedPoints=(divisions:number)=>curve.getSpacedPoints(divisions).map(p=>p.applyMatrix4(rope.matrixWorld));
    return {rgb:pixels.rgb,mask:pixels.mask,metadata:makeMetadata(sample,camera,worldCurve,pixels.maskPixels)};
  }
}
