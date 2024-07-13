import * as tf from '@tensorflow/tfjs';
import { Display } from './display';
import { Trainer } from './train';
import { VAEModel } from './model';
import { TrainingData } from './dataProcessor';

// class ImageInput {
//     private inputtedImages: FileList;

// }

export class Generator {
    private _latentDimSize: number | null = null;
    private _originalImageDimension: number | null = null;
    private _trainingData: TrainingData = new TrainingData();
    private decoder: tf.LayersModel | null = null;
    private display: Display = new Display();

    constructor () {

    }

    public set latentDimSize(latentDimSize: number) {
        if (latentDimSize <= 0) {
            throw new Error("Latent dimension size cannot be negative or zero.");
        }

        this._latentDimSize = latentDimSize;
    }

    public set originalImageDimension(originalImageDimension: number) {
        if (originalImageDimension < 0) {
            throw new Error("image dimension size cannot be negative or zero.");
        }

        this._originalImageDimension = originalImageDimension;
    }

    public get trainingData(): TrainingData {
        return this._trainingData;
    }

    public async loadPreloadedDecoder(preloadedDecoderPath: string = "./tf-models/32x32/model.json"): Promise<void> {
        const decoderModel = await tf.loadLayersModel(preloadedDecoderPath);

        console.log("problem is above...")

        // This is hard coded - depends on the preloaded model and how it was set up! For this model, the latent dim size was 16
        this.latentDimSize = 32;
        this.originalImageDimension = 32;
        this.decoder = decoderModel;
    }

    public async createNewModel(originalImageDimension: number, intermediateLayerDimensions: number[], latentDimension: number, epochs: number, batchSize: number): Promise<void> {
        const newModel: VAEModel = new VAEModel(originalImageDimension, intermediateLayerDimensions, latentDimension);

        // Train new model
        await Trainer.trainVAEModel(epochs, batchSize, newModel, this._trainingData);

        // Now set new decoder
        this.latentDimSize = latentDimension;
        this.originalImageDimension = originalImageDimension;
        this.decoder = newModel.decoderModel.model;
    }

    public async generatePiece(): Promise<void> {
        if (!this.decoder || !this._latentDimSize || !this._originalImageDimension) {
            throw new Error("Can't generate new piece when model has not been loaded to the Generator class.");
        }

        this.display.displayRandomlyGeneratedPiece(this.decoder, this._latentDimSize, this._originalImageDimension)
    }
}

export class OptionsManager {
    private generator: Generator;
    private imageDimensionInput: HTMLInputElement = document.getElementById("image-dimension-input") as HTMLInputElement;
    private nodeLayersInput: HTMLInputElement = document.getElementById("node-layer-input") as HTMLInputElement;
    private latentDimensionInput: HTMLInputElement = document.getElementById("latent-dimension-input") as HTMLInputElement;
    private trainingAmountInput: HTMLInputElement = document.getElementById("training-amount-input") as HTMLInputElement;
    private batchSizeInput: HTMLInputElement = document.getElementById("batch-size-input") as HTMLInputElement;
    private imageDropZone: HTMLDivElement = document.getElementById("add-extra-images-dropzone") as HTMLDivElement;
    private fileInput: HTMLInputElement = document.getElementById("file-input") as HTMLInputElement;
    private createNewModelButton: HTMLButtonElement = document.getElementById("generate-model-button") as HTMLButtonElement;
    private generalOptionsPrompt: HTMLParagraphElement = document.getElementById("general-options-prompt") as HTMLParagraphElement;
    private imageDropZonePrompt: HTMLParagraphElement = document.getElementById("image-drop-zone-prompt") as HTMLParagraphElement;

    constructor (generator: Generator) {
        this.generator = generator;
        this.addDropImagesFunctionality();
        this.addNewModelGenerationFunctionality();
    }

    private sendOptionError(errorMessage: string, optionElement: HTMLElement): void {
        // Send error message
        this.generalOptionsPrompt.innerText = errorMessage;

        // Play error animation
        optionElement.classList.add("error-animation-option-location");
        this.generalOptionsPrompt.classList.add("error-prompt");
        this.createNewModelButton.classList.add("error-prompt");

        // Revert animation and error message
        setTimeout(() => {
            this.generalOptionsPrompt.innerText = "🎯 Adjust some options and click generate to train your own model! Then start generating above! We provide you with 80 preset images so you don't need to add images if you want!";

            optionElement.classList.remove("error-animation-option-location");
            this.generalOptionsPrompt.classList.remove("error-prompt");
            this.createNewModelButton.classList.remove("error-prompt");
        }, 4000)
    }

    private addDropImagesFunctionality(): void {
        this.imageDropZone.addEventListener("click", () => {
            this.fileInput.click();
        })

        this.fileInput.addEventListener("change", async () => {
            const files: FileList | null = this.fileInput.files;

            if (!files) {
                return;
            }

            // Let user know that their files were inputted
            this.imageDropZonePrompt.innerText = "Awesome! You just inputted " + files.length + " files. We recommend around a 100 images for a good product. Also, don't forget to click on 'Generate New Model.'";
            
            // Play updated information animation
            this.imageDropZone.classList.add("updated");
            setTimeout(() => { this.imageDropZone.classList.remove("updated") }, 500);
        })
    }

    private addNewModelGenerationFunctionality(): void {
        this.createNewModelButton.addEventListener("click", async () => {
            const epochs: number = Math.abs(parseInt(this.trainingAmountInput.value));
            const batchSize: number = Math.abs(parseInt(this.batchSizeInput.value));
            const imageDimension: number = Math.abs(parseInt(this.imageDimensionInput.value));
            const latentDimension: number = Math.abs(parseInt(this.latentDimensionInput.value));
            const layers: number[] = this.nodeLayersInput.value.split(',').map(str => Math.abs(parseInt(str.trim())));
            const userImages: FileList | null = this.fileInput.files;

            if (isNaN(epochs) || epochs <= 0) {
                this.sendOptionError("Must atleast train once, the training amount cannot be 0!", this.trainingAmountInput);
                return;
            } 

            // Check if our current image database supports the batch size given
            if (isNaN(batchSize) || batchSize === 0 || userImages && userImages.length !== 0 && batchSize >= userImages.length || batchSize >= this.generator.trainingData.imageData.length) {
                this.sendOptionError("Batch size must not be 0 or exceed the amount of images given!", this.batchSizeInput);
                return;
            }

            if (isNaN(imageDimension) || imageDimension === 0) {
                this.sendOptionError("Image dimension can't be zero... 😭😭😭😭", this.imageDimensionInput);
                return;
            }

            if (isNaN(latentDimension) || latentDimension === 0) {
                this.sendOptionError("Latent dimension can't be zero, that's where we extract your data :)", this.latentDimensionInput);
                return;
            }

            // Check if any layer number given is not a number, and if so, return :)
            if (layers.some(Number.isNaN) || layers.some(layer => layer === 0)) {
                this.sendOptionError("Make sure all layers are valid numbers and seperated by commas! You're the best!", this.nodeLayersInput);
                return;
            }

            // Load new model
            this.createNewModelButton.innerText = "Loading... uh.. 👉👈";

            // IMPORTANT HACK: Use setTimeout to update the DOM before creating and training the new model - otherwise it would wait for the await functions... I know its ugly, but WE DON'T HAVE TIME RAJ
            setTimeout(async () => {
                if (!userImages) {
                    return;
                }

                if (userImages.length === 0) {
                    await this.generator.trainingData.loadPresetData(imageDimension);
                } else {
                    await this.generator.trainingData.loadCustomData(userImages, imageDimension);
                }

                // Now with the loaded data form the user, and their options, train the model! Yay!!!!!
                await this.generator.createNewModel(imageDimension, layers, latentDimension, epochs, batchSize);
                this.createNewModelButton.innerText = "Generate New Model";
            });
        })
    }
}

export class GenerateArtManager {
    private generator: Generator;
    private generatePieceButton: HTMLButtonElement = document.getElementById("generate-piece-button") as HTMLButtonElement;

    constructor (generator: Generator) {
        this.generator = generator;
        this.loadPreexistingModel(); // Give the generator a preexisting model to start with, training can come later :)
        this.addGenerateFunctionality();
    }

    private loadPreexistingModel(): void {
        this.generator.loadPreloadedDecoder("./tf-models/32x32/model.json");
    }

    private addGenerateFunctionality(): void {
        this.generatePieceButton.addEventListener("click", async () => {
            await this.generator.generatePiece();
        })
    }
}

export class DownloadGeneratedArtManager {
    private smallPreview: HTMLCanvasElement = document.getElementById("small-preview") as HTMLCanvasElement;
    private downloadGeneratedArtButton: HTMLButtonElement = document.getElementById("download-generated-art-button") as HTMLButtonElement;
    
    constructor () {
        this.addDownloadFunctionality();
    }

    private addDownloadFunctionality(): void {
        this.downloadGeneratedArtButton.addEventListener("click", () => {
            console.log(this.smallPreview.width, this.smallPreview.height)
            const dataURL = this.smallPreview.toDataURL("image/png");
            const link = document.createElement("a");
            link.href = dataURL;
            link.download = "canvas-image.png";
            link.click();
        })
    }
}