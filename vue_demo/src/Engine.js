// Copyright 2021 Ivan Alles. See also the LICENSE file. 

// eslint-disable no-unused-vars

import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

let MODEL_URL = '/prediction_model.tfjs/model.json';

function padSize(size, padTo) {
  if (size % padTo == 0)
    return 0;
  return padTo - size % padTo;
}

function makeInput(image, maxSize=511, padTo=32) {
  let sizeX = image.shape[1];
  let sizeY = image.shape[0];
  const scaleX = maxSize / sizeX;
  const scaleY = maxSize / sizeY;
  let scale = 0;
  if (scaleX < 1 || scaleY < 1) {
    if (scaleX < scaleY) {
      scale = scaleX;
      sizeX = maxSize;
      sizeY = Math.floor(sizeY * scale);
    }
    else {
      scale = scaleY;
      sizeX = Math.floor(sizeX * scale);
      sizeY = maxSize;
    }
    // console.log('sizeX, sizeY', sizeX, sizeY);
    image = tf.image.resizeBilinear(image, [sizeY, sizeX]);
  }
  const padX = padSize(sizeX, padTo);
  const padY = padSize(sizeY, padTo);
  if(padX != 0 || padY != 0) {
    image = tf.pad(image, [[0, padY], [0, padX], [0, 0]])
  }
  // RGB -> BGR, add batch dimension.
  image = tf.expandDims(tf.reverse(image, 2), 0);
  return {
    image,
    scale
  };  
}

class Localizer {
  constructor(logger) {
    this.logger = logger;
    this.models = {};
  }

  async init(onProgress) {
    let url = MODEL_URL;
    if (process.env.NODE_ENV === 'production' ) {
      url = '/preference-model' + url;      
    }

    console.log(`Loading model ${url} ...`)
    this.model = await loadGraphModel(url, {onProgress: fraction => {
      let progress = Math.round(fraction * 100);
      onProgress(`Loading neural network: ${progress}%`);
    }});
    console.log(`Model loaded`);
  }

  async predict(image) {
    let input = null;
    let output = null;
    try {
      // console.log('image', image)
      const input = tf.tidy(() => makeInput(image, 512, 32));
      const output = await this.model.executeAsync({'image': input.image});
      // console.log('prediction', prediction);
      const objects = await output.data()

      let canvas = document.createElement('canvas');
      await tf.browser.toPixels(image, canvas);
      
      for(let o = 0; o < objects.length; o += 5) {
        const x = objects[o] * 8 / input.scale
        const y = objects[o + 1] * 8 / input.scale
        const angle = objects[o + 2]
        // console.log('x, y, angle', x, y, angle);

        var ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 3;
        ctx.setTransform(Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle), x, y);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(44, 0);
        ctx.lineTo(34, -10);
        ctx.moveTo(44, 0);
        ctx.lineTo(34, 10);
        ctx.stroke();

      }
      
      // let output = tf.clipByValue(prediction[2], 0, 1);
      // output = tf.squeeze(tf.unstack(output, 4)[0]);
      // console.log('output', output)

      return canvas.toDataURL('image/png');
    }
    catch(error) {
      this.logger.logException('Localizer.predict', error);
    }
    finally {
      tf.dispose(input);
      tf.dispose(output);
    }
    return null; 
  }
}

/**
 * Combines alogrithms to implement application logic.
 * Converts the data between UI (plain javascript data) to internal representations (tf.tensor).
 */
export class Engine {
  constructor(logger) {
    this.logger = logger;
  }

  async init(onProgress) {
    // Do tf initialization here, before any usage of it.
    await tf.ready();

    console.log('Initial tf.ENV.features', tf.ENV.features);
    
    // Switch off the convolution algo conv2dWithIm2Row() with very high memory requirements.
    tf.ENV.set('WEBGL_CONV_IM2COL', false)

    // Use 16-bit
    tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true)
    tf.ENV.set('WEBGL_RENDER_FLOAT32_ENABLED', false)

    // Repeat just in case the settings above require some lengthy initialization.
    await tf.ready();

    console.log('New tf.ENV.features', tf.ENV.features);


    if (process.env.NODE_ENV === 'production' ) {
      console.log('Production mode');
      tf.enableProdMode();
    }
    else {
      console.log('Development mode');
    }

    this.localizer = new Localizer(this.logger);

    await this.localizer.init(onProgress);
  }

  async predict(image) {
    // console.log('tf.memory', tf.memory());
    let imageTensor = null;
    try {
      let imageTensor = tf.tidy(() =>  
        tf.div(tf.cast(tf.browser.fromPixels(image), 'float32'), 255));
      return await this.localizer.predict(imageTensor);
    }
    finally {
      tf.dispose(imageTensor);
    }
  }

}

// Export for tests only.
export const testables = {padSize, makeInput};