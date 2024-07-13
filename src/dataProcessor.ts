import * as tf from '@tensorflow/tfjs';

export class TrainingData {
  private _imageData: tf.Tensor[] = new Array(80); // 80 because we have preloaded data to fall back on if the user doesn't input anything

  constructor () {
    
  }

  public get imageData(): tf.Tensor[] {
    return this._imageData;
  }

  private async convertFileObjectToURLs(files: FileList): Promise<string[]> {
    const promises = Array.from(files).map((file: File) => {
      return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event: ProgressEvent<FileReader>) => { resolve(event.target?.result as string); };
        reader.onerror = reject;
        reader.readAsDataURL(file); // Start reading the file
      })
    })

    return Promise.all(promises);
  }

  private async loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous'; // This is needed if you're loading images from a different origin
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  private async processImage(image: HTMLImageElement, imageReshapeDimension: number): Promise<tf.Tensor<tf.Rank>> {
    const tensor = tf.browser.fromPixels(image, 3); // Convert to grayscale
    const resized = tf.image.resizeBilinear(tensor, [imageReshapeDimension, imageReshapeDimension]); // Resize
    const normalized = resized.div(255.0); // Normalize pixel values to [0, 1]
    return normalized.expandDims(0); // Add a batch dimension
  }

  public async loadPresetData(imageReshapeDimension: number): Promise<void> {
    // Load 80 preset images with their size in the project
    const images = new Array(80);
  
    for (let i = 0; i < 80; i++) {
      const file = "/images/trainingData/" + i + ".jpg";
      const image = await this.loadImage(file);
      const processedImage = await this.processImage(image, imageReshapeDimension);
      images[i] = (processedImage);
    }
  
    this._imageData = images;
  }

  public async loadCustomData(files: FileList, imageReshapeDimension: number): Promise<void> {
    const userImageURLs: string[] = await this.convertFileObjectToURLs(files);
    const images = new Array(userImageURLs.length);
  
    for (let i = 0; i < userImageURLs.length; i++) {
      const file = userImageURLs[i];
      const image = await this.loadImage(file);
      const processedImage = await this.processImage(image, imageReshapeDimension);
      images[i] = (processedImage);
    }
    
    this._imageData = images;
  }
}