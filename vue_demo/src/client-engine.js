// Copyright 2016-2020 Ivan Alles. See also the LICENSE file. 

// eslint-disable no-unused-vars

import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

let MODEL_URL = '/prediction_model.tfjs/model.json';

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
    let result = null;
    try {
      console.log('image', image)
      const prediction = this.model.predict({'image': image});
      // console.log('prediction', prediction);
      const output = tf.clipByValue(tf.squeeze(prediction[3]), 0, 1);
      console.log('output', output)

      let canvas = document.createElement('canvas');
      await tf.browser.toPixels(output, canvas);
      // Use JPEG compression as potentially more compact.
      // The performance with the default quality is better than PNG.
      result = canvas.toDataURL('image/jpg');
      return result;
    }
    catch(error) {
      this.logger.logException('Generator.generate', error);
    }
    finally {
      tf.dispose(result);
    }
    return result; 
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
        tf.div(tf.cast(tf.expandDims(tf.browser.fromPixels(image), 0), 'float32'), 255));
      return await this.localizer.predict(imageTensor);
    }
    finally {
      tf.dispose(imageTensor);
    }
  }

}

// TODO(ia): clean-up
// Export for tests only.
// export const testables = {PreferenceModel, sphericalToCartesian, cartesianToSpherical, scaledDirichlet};