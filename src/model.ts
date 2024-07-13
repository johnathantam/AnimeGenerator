import * as tf from '@tensorflow/tfjs';
import { ZLayer } from './customLayers';

export interface Model {
    originalImageDimension: number,
    intermediateLayerDimensions: number[],
    latentDimension: number,
    model: tf.LayersModel,
}

class EncoderModel implements Model {
    private _originalImageDimension: number = 28;
    private _intermediateLayerDimensions: number[] = [64, 32];
    private _latentDimension: number = 2;
    private _model: tf.LayersModel;

    constructor (originalImageDimension: number, intermediateLayerDimensions: number[], latentDimension: number) {
        this.originalImageDimension = originalImageDimension;
        this.intermediateLayerDimensions = intermediateLayerDimensions;
        this.latentDimension = latentDimension;
        this._model = this.initializeEncoder();
    }

    public set originalImageDimension(originalImageDimension: number) {
        if (originalImageDimension <= 0) {
            throw new Error("Image dimension cannot be negative or zero.");
        }

        this._originalImageDimension = originalImageDimension;
    }

    public set intermediateLayerDimensions(intermediateLayerDimensions: number[]) {
        for (let i = 0; i < intermediateLayerDimensions.length; i++) {
            if (intermediateLayerDimensions[i] < 0) {
                throw new Error("Can't have negative units / nodes in a layer!");
            }
        }

        this._intermediateLayerDimensions = intermediateLayerDimensions;
    }

    public set latentDimension(latentDimension: number) {
        if (latentDimension <= 0) {
            throw new Error("Latent dimension cannot be negative or zero.");
        }

        this._latentDimension = latentDimension;
    }

    public get originalImageDimension(): number {
        return this._originalImageDimension;
    }

    public get intermediateLayerDimensions(): number[] {
        return this._intermediateLayerDimensions;
    }

    public get latentDimension(): number {
        return this._latentDimension;
    }

    public get model(): tf.LayersModel{
        return this._model;
    }

    private initializeEncoder(): tf.LayersModel {
        const inputs: tf.SymbolicTensor = tf.input({shape: [this._originalImageDimension], name: 'encoder_input'}) as tf.SymbolicTensor;
        
        // Add intermediate layers
        let x = inputs;

        for (let i = 0; i < this._intermediateLayerDimensions.length; i++) {
            x = tf.layers.dense({ units: this._intermediateLayerDimensions[i], activation: 'relu' }).apply(x) as tf.SymbolicTensor;
        }

        // Add final layers or z layers which are the layers that store the compressed data 
        const zMean: tf.SymbolicTensor = tf.layers.dense({ units: this._latentDimension, name: 'z_mean' }).apply(x) as tf.SymbolicTensor;
        const zLogVar: tf.SymbolicTensor = tf.layers.dense({ units: this._latentDimension, name: 'z_log_var' }).apply(x) as tf.SymbolicTensor;
        const z: tf.SymbolicTensor = new ZLayer({ name: 'z', outputShape: [this._latentDimension] }).apply([zMean, zLogVar]) as tf.SymbolicTensor;

        // Create encoder - the three outputs of zMean, zLogVar are used for training while z is the actual compressed representation from the autoencoder of a image
        const encoder: tf.LayersModel = tf.model({ inputs: inputs, outputs: [zMean, zLogVar, z], name: 'encoder', });
        return encoder;
    }
}

class DecoderModel implements Model {
    private _originalImageDimension: number = 28;
    private _intermediateLayerDimensions: number[] = [64, 32];
    private _latentDimension: number = 2;
    private _model: tf.LayersModel;

    constructor (originalImageDimension: number, intermediateLayerDimensions: number[], latentDimension: number) {
        this.originalImageDimension = originalImageDimension;
        this.intermediateLayerDimensions = intermediateLayerDimensions;
        this.latentDimension = latentDimension;
        this._model = this.initializeDecoder();
    }

    public set originalImageDimension(originalImageDimension: number) {
        if (originalImageDimension <= 0) {
            throw new Error("Image dimension cannot be negative or zero.");
        }

        this._originalImageDimension = originalImageDimension;
    }

    public set intermediateLayerDimensions(intermediateLayerDimensions: number[]) {
        for (let i = 0; i < intermediateLayerDimensions.length; i++) {
            if (intermediateLayerDimensions[i] < 0) {
                throw new Error("Can't have negative units / nodes in a layer!");
            }
        }

        this._intermediateLayerDimensions = intermediateLayerDimensions;
    }

    public set latentDimension(latentDimension: number) {
        if (latentDimension <= 0) {
            throw new Error("Latent dimension cannot be negative or zero.");
        }

        this._latentDimension = latentDimension;
    }

    public get originalImageDimension(): number {
        return this._originalImageDimension;
    }

    public get intermediateLayerDimensions(): number[] {
        return this._intermediateLayerDimensions;
    }

    public get latentDimension(): number {
        return this._latentDimension;
    }

    public get model(): tf.LayersModel {
        return this._model;
    }

    private initializeDecoder(): tf.LayersModel {
        const input: tf.SymbolicTensor = tf.input({shape: [this._latentDimension]}) as tf.SymbolicTensor;
        
        // Add intermediate layers
        let y = input;

        for (let i = 0; i < this._intermediateLayerDimensions.length; i++) {
            y = tf.layers.dense({ units: this._intermediateLayerDimensions[i], activation: 'relu' }).apply(y) as tf.SymbolicTensor;
        }

        // Add final layers or z layers which are the layers that store the compressed data 
        const outputLayer: tf.SymbolicTensor = tf.layers.dense({ units: this._originalImageDimension, activation: 'sigmoid' }).apply(y) as tf.SymbolicTensor;

        // Create decoder
        const decoder: tf.LayersModel = tf.model({inputs: input, outputs: outputLayer});
        return decoder;
    }
}

export class VAEModel implements Model {
    private _originalImageDimension: number = 784;
    private _intermediateLayerDimensions: number[] = [64, 32];
    private _latentDimension: number = 2;
    private _encoderModel: EncoderModel;
    private _decoderModel: DecoderModel;
    private _model: tf.LayersModel;

    constructor (originalImageDimension: number, intermediateLayerDimensions: number[], latentDimension: number) {
        this.originalImageDimension = originalImageDimension * originalImageDimension * 3;
        this.intermediateLayerDimensions = intermediateLayerDimensions;
        this.latentDimension = latentDimension;
        this._encoderModel = new EncoderModel(this._originalImageDimension, this._intermediateLayerDimensions, this._latentDimension);
        this._decoderModel = new DecoderModel(this._originalImageDimension, this._intermediateLayerDimensions, this._latentDimension);
        this._model = this.initializeVAEModel();
    }

    public set originalImageDimension(originalImageDimension: number) {
        if (originalImageDimension <= 0) {
            throw new Error("Image dimension cannot be negative or zero.");
        }

        this._originalImageDimension = originalImageDimension;
    }

    public set intermediateLayerDimensions(intermediateLayerDimensions: number[]) {
        for (let i = 0; i < intermediateLayerDimensions.length; i++) {
            if (intermediateLayerDimensions[i] < 0) {
                throw new Error("Can't have negative units / nodes in a layer!");
            }
        }

        this._intermediateLayerDimensions = intermediateLayerDimensions;
    }

    public set latentDimension(latentDimension: number) {
        if (latentDimension <= 0) {
            throw new Error("Latent dimension cannot be negative or zero.");
        }

        this._latentDimension = latentDimension;
    }

    public get originalImageDimension(): number {
        return this._originalImageDimension;
    }

    public get intermediateLayerDimensions(): number[] {
        return this._intermediateLayerDimensions;
    }

    public get latentDimension(): number {
        return this._latentDimension;
    }

    public get encoderModel(): EncoderModel {
        return this._encoderModel
    }

    public get decoderModel(): DecoderModel  {
        return this._decoderModel;
    }

    public get model(): tf.LayersModel {
        return this._model;
    }

    public getVAELoss(inputs: tf.Tensor, outputs: tf.Tensor[]): tf.Scalar {
        return tf.tidy(() => {
            const originalDim: number = inputs.shape[1] || this._originalImageDimension * this._originalImageDimension * 3;
            const decoderOutput: tf.Tensor<tf.Rank> = outputs[0];
            const zMean: tf.Tensor<tf.Rank> = outputs[1];
            const zLogVar: tf.Tensor<tf.Rank> = outputs[2];
        
            // First we compute a 'reconstruction loss' terms. The goal of minimizing
            // this term is to make the model outputs match the input data.
            //const reconstructionLoss = tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim);
        
            // binaryCrossEntropy can be used as an alternative loss function
            const reconstructionLoss: tf.Tensor<tf.Rank> = tf.metrics.binaryCrossentropy(inputs, decoderOutput).mul(originalDim);
        
            // Next we compute the KL-divergence between zLogVar and zMean, minimizing
            // this term aims to make the distribution of latent variable more normally
            // distributed around the center of the latent space.
            let klLoss: tf.Tensor<tf.Rank> = zLogVar.add(1).sub(zMean.square()).sub(zLogVar.exp());
            klLoss = klLoss.sum(-1).mul(-0.5);
        
            return reconstructionLoss.add(klLoss).mean();
        });
    }

    public initializeVAEModel(): tf.LayersModel {
        const inputs: tf.SymbolicTensor[] = this._encoderModel.model.inputs as tf.SymbolicTensor[];
        const encoderOutputs: tf.SymbolicTensor[] = this._encoderModel.model.apply(inputs) as tf.SymbolicTensor[];
        const encoded: tf.SymbolicTensor = encoderOutputs[2] as tf.SymbolicTensor;
        const decoderOutput: tf.SymbolicTensor = this._decoderModel.model.apply(encoded) as tf.SymbolicTensor;

        const model: tf.LayersModel = tf.model({
            inputs: inputs,
            outputs: [decoderOutput, ...encoderOutputs],
            name: 'vae_mlp',
        })

        return model;
    }
}