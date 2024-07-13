"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const tf = __importStar(require("@tensorflow/tfjs")); // Use '@tensorflow/tfjs-node' for Node.js compatibility
// // Define a simple model
// const model = tf.sequential();
// console.log(model);
// Load MNIST dataset (this is just an example, MNIST data loading in Node.js is different)
// For real MNIST data, you would need to load it differently in Node.js, possibly from files or a database
// For demonstration, we'll use random data to simulate MNIST
// Generate random data for demonstration (replace this with actual MNIST loading)
const NUM_EXAMPLES = 10000;
const INPUT_SHAPE = [784]; // 28x28 flattened
const xTrain = tf.randomNormal([NUM_EXAMPLES, ...INPUT_SHAPE]);
const xTest = tf.randomNormal([100, ...INPUT_SHAPE]);
// Define the autoencoder model
const autoencoder = tf.sequential();
// Encoder
autoencoder.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: INPUT_SHAPE }));
// Decoder
autoencoder.add(tf.layers.dense({ units: 784, activation: 'sigmoid' }));
// Compile the autoencoder
autoencoder.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
// Train the autoencoder
function train() {
    return __awaiter(this, void 0, void 0, function* () {
        yield autoencoder.fit(xTrain, xTrain, {
            epochs: 20,
            batchSize: 256,
            validationData: [xTest, xTest],
        });
        console.log('Training complete.');
        // Example of using the trained model to encode and decode data
        const encoded = autoencoder.predict(xTest);
        const decoded = encoded;
        // Example of saving the model
        yield autoencoder.save('file://./my_model');
        // Example of loading the model
        const loadedModel = yield tf.loadLayersModel('file://./my_model/model.json');
    });
}
train().catch(console.error);
