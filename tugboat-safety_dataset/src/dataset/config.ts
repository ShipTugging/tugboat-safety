export const GENERATOR_VERSION = 'towline-1.0.0';
export const datasetConfig = {
  datasetSize: 500, seed: 1043, imageWidth: 1280, imageHeight: 720,
  centerlinePoints: 50,
  split: { train: .7, val: .2, test: .1 },
  towline: { sagRange: [.02, 2.0], radiusRange: [.06, .11], brightnessRange: [.35, .85] },
  merchantVessel: { distanceRange: [16, 28], lateralOffsetRange: [8, 18], headingRange: [-.06, .06] },
  camera: { mount: [-2.7, 4.8, 1], pitchRange: [-.025, .025], rollRange: [-.025, .025], yawRange: [-.025, .025], fovRange: [64, 72] },
  environment: { lightIntensityRange: [.8, 1.2], sunXRange: [-100, 80], sunYRange: [90, 180], sunZRange: [20, 100], brightnessRange: [.9, 1.15], fogRange: [.0002, .003], waveStrengthRange: [.4, .9], timeRange: [60, 120] },
  validation: { minMaskPixels: 12, minInFrameFraction: .7, minVisibleFraction: .6, minProjectedChordPixels: 60, maxAttemptsPerSample: 20 },
};
export type DatasetConfig = typeof datasetConfig;
export type Split = 'train' | 'val' | 'test';
export function validateConfig(c: DatasetConfig) {
  for (const [key, min, max] of [['datasetSize',1,5000],['seed',0,4294967295],['imageWidth',64,4096],['imageHeight',64,4096],['centerlinePoints',3,500]] as const) {
    if (!Number.isInteger(c[key]) || c[key] < min || c[key] > max) throw new Error(`Invalid ${key}: expected integer ${min}–${max}`);
  }
  if (Object.values(c.split).some(v=>!Number.isFinite(v)||v<0) || Math.abs(Object.values(c.split).reduce((a,b)=>a+b,0)-1)>1e-9) throw new Error('Split ratios must sum to 1');
  for (const section of [c.towline,c.merchantVessel,c.camera,c.environment]) for (const [key,value] of Object.entries(section)) {
    if (key.endsWith('Range') && (value.length!==2 || !value.every(Number.isFinite) || value[0]>value[1])) throw new Error(`Invalid range ${key}`);
  }
  if(c.towline.sagRange[0]<0 || c.towline.radiusRange[0]<=0 || c.camera.fovRange[0]<=0 || c.camera.fovRange[1]>=150) throw new Error('Invalid rope or FOV range');
  if(c.camera.mount.length!==3 || !c.camera.mount.every(Number.isFinite)) throw new Error('Invalid camera mount');
  const v=c.validation;
  if(!Number.isInteger(v.maxAttemptsPerSample)||v.maxAttemptsPerSample<1||v.maxAttemptsPerSample>100 || !Number.isInteger(v.minMaskPixels)||v.minMaskPixels<1 || !Number.isFinite(v.minProjectedChordPixels)||v.minProjectedChordPixels<=0 || [v.minInFrameFraction,v.minVisibleFraction].some(n=>!Number.isFinite(n)||n<0||n>1)) throw new Error('Invalid validation thresholds');
}
