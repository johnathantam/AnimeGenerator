import * as tf from '@tensorflow/tfjs';
import { TrainingData } from './dataProcessor';
import { VAEModel } from './model';

export class Trainer {
    public static async trainVAEModel(epochs: number, batchSize: number, vaeModel: VAEModel, trainingData: TrainingData): Promise<void> {
        const images: tf.Tensor[] = trainingData.imageData;

        if (images === null) {
            throw new Error("Training data was not initialized before it was passed into the trainer.");
        }

        if (batchSize > images.length || batchSize < 0) {
            throw new Error("Batch size doesn't match data");
        }

        if (epochs < 0) {
            throw new Error("Can't train negative times with negative epochs");
        }

        for (let i = 0; i < epochs; i++) {
            // Shuffle the array of images randomly
            tf.util.shuffle(images);
        
            // Take a random subset of the specified size
            const randomSubset = images.slice(0, batchSize);
        
            const xTrain = tf.concat(randomSubset);
            const xTrainReshaped = xTrain.reshape([xTrain.shape[0], vaeModel.originalImageDimension]);
        
            // Compile the VAE model with the custom loss function
            const optimizer = tf.train.adam();
        
            optimizer.minimize((): tf.Scalar => {
              const outputs = vaeModel.model.apply(xTrainReshaped) as tf.Tensor<tf.Rank>[];
              const loss = vaeModel.getVAELoss(xTrainReshaped, outputs);
        
              // Log the loss - little easter egg I guess :)
              console.log(`Epoch ${i + 1}: Loss = ${loss.dataSync()}`);
        
              return loss;
            })
        
            tf.dispose([xTrainReshaped, xTrain]);
        }
    }
}