import * as tf from '@tensorflow/tfjs';

export class Display {
  private bigPreview: HTMLCanvasElement = document.getElementById("big-preview") as HTMLCanvasElement;
  private bigPreviewContext: CanvasRenderingContext2D = this.bigPreview.getContext("2d") as CanvasRenderingContext2D;
  private smallPreview: HTMLCanvasElement = document.getElementById("small-preview") as HTMLCanvasElement;
  private smallPreviewContext: CanvasRenderingContext2D = this.smallPreview.getContext("2d") as CanvasRenderingContext2D;
  private tinyPreview: HTMLCanvasElement = document.getElementById("tiny-preview") as HTMLCanvasElement;
  private tinyPreviewContext: CanvasRenderingContext2D = this.tinyPreview.getContext("2d") as CanvasRenderingContext2D;

  constructor () {
    this.adjustDisplayWidths()
  }

  private adjustDisplayWidths(): void {
    // Canvas' in the project are stylized with custom css widths and heights, so when we initialize, we adjust the canvas widths and heights accordingly to the css!
    const smallPreviewPixelSize: number = parseFloat(window.getComputedStyle(this.smallPreview).width);
    const bigPreviewPixelSize: number = parseFloat(window.getComputedStyle(this.bigPreview).width);
    const tinyPreviewPixelSize: number = parseFloat(window.getComputedStyle(this.tinyPreview).width);

    this.tinyPreview.width = tinyPreviewPixelSize;
    this.tinyPreview.height = tinyPreviewPixelSize;
    this.tinyPreviewContext.imageSmoothingEnabled = false;

    this.smallPreview.width = smallPreviewPixelSize;
    this.smallPreview.height = smallPreviewPixelSize;
    this.smallPreviewContext.imageSmoothingEnabled = false;
    
    this.bigPreview.width = bigPreviewPixelSize;
    this.bigPreview.height = bigPreviewPixelSize;
    this.bigPreviewContext.imageSmoothingEnabled = false;
  }

  private async generate(decoderModel: tf.LayersModel, latentDimSize: number): Promise<Float32Array> {
    const mean = 0;
    const std = 1.0;
    const targetZ = tf.randomNormal([1, latentDimSize], mean, std);
    const generated = (decoderModel.predict(targetZ)) as tf.Tensor<tf.Rank>;
    
    return generated.dataSync() as Float32Array;
  }
    
  private visualizeImage(imageData: Float32Array, imageDimensions: number): void {
    // Create a template canvas with the dimensions of the image
    const templateCanvas = document.createElement('canvas') as HTMLCanvasElement;
    const templateContext = templateCanvas.getContext('2d') as CanvasRenderingContext2D;
    templateCanvas.width = imageDimensions;
    templateCanvas.height = imageDimensions;

    // Normalize the pixel values to [0, 255]
    const pixelData = imageData.map(value => Math.round(value * 255));
  
    // Create ImageData object
    const imgData = this.bigPreviewContext.createImageData(imageDimensions, imageDimensions);
  
    // Set the image data
    for (let i = 0; i < imgData.data.length; i += 4) {
      const j = i / 4;
      imgData.data[i + 0] = pixelData[j * 3 + 0]; // R
      imgData.data[i + 1] = pixelData[j * 3 + 1]; // G
      imgData.data[i + 2] = pixelData[j * 3 + 2]; // B
      imgData.data[i + 3] = 255; // Alpha
    }
  
    // Draw ImageData onto the template canvas with proper 
    templateContext.putImageData(imgData, 0, 0);
    this.bigPreviewContext.drawImage(templateCanvas, 0, 0, this.bigPreview.width, this.bigPreview.height);
    this.smallPreviewContext.drawImage(templateCanvas, 0, 0, this.smallPreview.width, this.smallPreview.height);
    this.tinyPreviewContext.drawImage(templateCanvas, 0, 0, this.tinyPreview.width, this.tinyPreview.height);
  }

  public async displayRandomlyGeneratedPiece(decoder: tf.LayersModel, latentDimSize: number, imageDimensions: number): Promise<void> {
    const generatedImageData: Float32Array = await this.generate(decoder, latentDimSize);
    this.visualizeImage(generatedImageData, imageDimensions);
  }
}