import * as tf from '@tensorflow/tfjs';

export class ZLayer extends tf.layers.Layer {
    constructor(config: any) {
      super(config);
    }
  
    public computeOutputShape(inputShape: tf.Shape[]): tf.Shape {
      tf.util.assert(inputShape.length === 2 && Array.isArray(inputShape[0]), () => `Expected exactly 2 input shapes. But got: ${inputShape}`);
      return inputShape[0];
    }
  
    public call(inputs: tf.Tensor[], kwargs: any): tf.Tensor {
      const [zMean, zLogVar] = inputs;
      const batch: number = zMean.shape[0] || 8;
      const dim: number = zMean.shape[1] || 784;
  
      const mean = 0;
      const std = 1.0;
      // sample epsilon = N(0, I)
      const epsilon = tf.randomNormal([batch, dim], mean, std);
  
      // z = z_mean + sqrt(var) * epsilon
      return zMean.add(zLogVar.mul(0.5).exp().mul(epsilon));
    }
  
    static get className(): string {
      return 'ZLayer';
    }
  }