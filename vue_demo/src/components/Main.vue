<!-- Copyright 2021 Ivan Alles. See also the LICENSE file. -->

<template>
  <b-container>
    <div>
      <h1>Localizer</h1>
    </div>
    <template v-if="state === stateKind.WORKING">
      <canvas id="viewCanvas"></canvas>
    </template>
    <template v-if="state === stateKind.INIT">
      <h4>
        <b-spinner variant="secondary" label="Loading"></b-spinner>
        {{progressMessage}}
      </h4>
      <template v-if="isMobile">
        <p>Running this app on a mobile device can be slow or irresponsive. It works best on a desktop with an NVIDIA graphic card.</p>
      </template>
    </template>
    <template v-if="state === stateKind.ERROR">
      <!-- The practice have shown that in case of an error we cannot recover. Only reloading the page helps. -->
      <h4 class="error">
        <b-icon icon="exclamation-circle-fill" variant="danger"></b-icon>
        Error
      </h4>
      <p>
      This app works best on a desktop with an NVIDIA graphic card. Other devices may not be supported.
      </p>
      <b-button @click="reload()" variant="primary">
        <b-icon icon="bootstrap-reboot"></b-icon>
          Reload
        </b-button>
    </template>
  </b-container>
</template>

<script>

import { Engine } from '@/Engine'

const stateKind = {
    INIT: 'INIT',       // Loading models, etc.
    WORKING: 'WORKING', // Generating pictures
    EXIT: 'EXIT',       // App finished.
    ERROR: 'ERROR',     // Fatal error, cannot work.
}

class GoogleAnalyticsLogger {
  constructor(ga) {
    this.ga = ga;
  }

  log(category, action, label, value=1) {
    console.log(category, action, label, value);
    this.ga.event({
      eventCategory: category,
      eventAction: action,
      eventLabel: label,
      eventValue: value
    });
  }

  logException(action, exception, value=1) {
    console.error(exception);
    this.ga.event({
      eventCategory: 'LogError',
      eventAction: action,
      eventLabel: exception.stack,
      eventValue: value
    });
  }
}

export default {
  data() {
    return {
      state: stateKind.INIT,
      isMobile: false,
      progressMessage: 'Loading ...',
      tempResultPicture: null,
      camera: null,
      isVideoReady: false,
    };
  },
  computed: {
  },  

  methods: {
    /**
    * Generates pictures in the background.
    */
    async getPicturesTask() {
      try {
        await this.engine.init(message => {this.progressMessage = message;});
        this.progressMessage = 'Warming up ...';

        while(!this.isVideoReady) {
          await sleep(50);
        }
        console.log('isVideoReady state: ', this.isVideoReady);

        this.state = stateKind.WORKING;

        const bufferCanvas = document.createElement('canvas');
        bufferCanvas.setAttribute('width', this.camera.videoWidth);
        bufferCanvas.setAttribute('height', this.camera.videoHeight);

        while(this.state != stateKind.EXIT) {
          await sleep(50);
          if(!this.isActive) {
            continue;
          }
          
          const bufferContext = bufferCanvas.getContext("2d");

          // Flip camera image.
          bufferContext.translate(bufferCanvas.width, 0);
          bufferContext.scale(-1, 1);
          bufferContext.drawImage(this.camera, 0, 0);
          bufferContext.resetTransform();

          const objects = await this.engine.predict(bufferCanvas);

          // console.log('getting viewCanvas');
          const viewCanvas = document.getElementById('viewCanvas');
          // Make View canvas fit the screen and preserve the aspect ratio.
          viewCanvas.setAttribute('width', window.innerWidth);
          const viewScale = viewCanvas.width / bufferCanvas.width;
          viewCanvas.setAttribute('height', bufferCanvas.height * viewScale);

          const viewContext = viewCanvas.getContext("2d");
          viewContext.drawImage(bufferCanvas, 0, 0, bufferCanvas.width, bufferCanvas.height, 0, 0, viewCanvas.width, viewCanvas.height);

          const objScale = viewScale * this.engine.objectSize / 2;

          for(const o of objects) {
            // console.log('object', o);
            viewContext.strokeStyle = "#00FF00";
            const sa = Math.sin(o.angle) * objScale;
            const ca = Math.cos(o.angle) * objScale;
            viewContext.transform(ca, sa, -sa, ca, o.x * viewScale, o.y * viewScale);
            viewContext.beginPath();
            viewContext.moveTo(0, 0);
            viewContext.lineTo(1.05, 0);
            viewContext.moveTo(1, 0);
            viewContext.lineTo(0.8, -0.15);
            viewContext.moveTo(1, 0);
            viewContext.lineTo(0.8, 0.15);
            viewContext.resetTransform();
            viewContext.lineWidth = 3;
            viewContext.stroke();
          }
        }
      }
      catch(error) {
        this.state = stateKind.ERROR;
        this.logger.logException('Images.getPicturesTask.createPictures', error);
        return;
      }
    },

    reload() {
      location.reload();
    },

    onVideoReady() {
      console.log('Video ready.');
      this.isVideoReady = true;
    },

    async startVideo() {
      // this.camera = document.querySelector("#videoElement");
      this.camera = document.createElement("video");
      if (navigator.mediaDevices.getUserMedia) {
        this.camera.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
      }
      this.camera.onloadeddata = this.onVideoReady;

      this.camera.play();
    },    
  },

  created() {
    // Make globals accessible in Vue rendering code
    this.stateKind = stateKind;

    this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    this.logger = new GoogleAnalyticsLogger(this.$ga);
    this.isActive = true;
    this.engine = new Engine(this.logger);
  },

  mounted() {
    this.startVideo();
    this.getPicturesTask();
  },

  beforeDestroy () {
    this.state = stateKind.EXIT;
  },

  updated() {
  },

  watch: {
    $route(to, from) { // eslint-disable-line
      // Activate this component when the router points to it.
      this.isActive = to.name === 'Home';
    }
  },

};

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

</script>

<style scoped>

.error {
  color: var(--danger);
}


</style>
