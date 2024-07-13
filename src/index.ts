import { Generator, OptionsManager, GenerateArtManager, DownloadGeneratedArtManager } from './ui';

class App {
  private static _instance: App;
  private generator: Generator = new Generator();
  private optionManager: OptionsManager = new OptionsManager(this.generator);
  private generateArtManager: GenerateArtManager = new GenerateArtManager(this.generator);
  private downloadArtManager: DownloadGeneratedArtManager = new DownloadGeneratedArtManager();

  // Make the constructor private to prevent direct instantiation
  private constructor() {}

  public static get instance(): App {
    if (!App._instance) {
      App._instance = new App();
    }

    return App._instance;
  }

}

class Driver {
  constructor () {
    App.instance;
  }
}

new Driver();