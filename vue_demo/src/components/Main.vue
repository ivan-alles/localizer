<!-- Copyright 2021 Ivan Alles. See also the LICENSE file. -->

<template>
  <b-container>
    <template v-if="state === stateKind.WELCOME">
      <h1>Hand Detector</h1>
      <h4>Localizer demo app</h4>
      <div class='social-share'>
        <ShareNetwork
            network="Facebook"
            :url="shareUrl()"
            :title="shareTitle()"
          >
          <b-button variant="secondary">
            <font-awesome-icon :icon="['fab', 'facebook']" size="lg" ></font-awesome-icon>
          </b-button>
        </ShareNetwork>
        <!-- Twitter does not work with a local URL. -->
        <ShareNetwork
            network="Twitter"
            :url="shareUrl()"
            :title="shareTitle()"
          >
          <b-button variant="secondary">
            <font-awesome-icon :icon="['fab', 'twitter']" size="lg" ></font-awesome-icon>
          </b-button>
        </ShareNetwork>         
        <ShareNetwork
            network="VK"
            :url="shareUrl()"
            :title="shareTitle()"
          >
          <b-button variant="secondary">
            <font-awesome-icon :icon="['fab', 'vk']" size="lg" ></font-awesome-icon>
          </b-button>
        </ShareNetwork>   
        <!-- The URL will be inserted as plain text, so add a line break and a short description. -->
        <ShareNetwork
            network="Email"
            :url="shareUrl()"
            :title="shareTitle()"
            description="Check out the localizer demo app."
          >
          <b-button variant="secondary">
            <b-icon icon="envelope" ></b-icon>
          </b-button>
        </ShareNetwork>
      </div>
      <div class="youtube-super-container">
        <div class="youtube-container">
          <iframe class="youtube-video" src="https://www.youtube.com/embed/Pcbk_Hx1WpU" frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>                
        </div>
      </div>
      <div class="main-text">
        <p>This app detects your hands on a live camera video. It is powered by the <b>Localizer</b>: a neural network for object detection.</p>
        <p><a href="https://ivan-alles.github.io/localizer/">Get more</a> on GitHub: source code, examples, hands-on python app.</p>
      </div>
      <b-button @click="startDemo()" variant="primary">
        <b-icon icon="camera-video" ></b-icon>
        Start
      </b-button>
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
    <template v-if="state === stateKind.WORKING">
      <div id="viewContainer">
        <canvas id="viewCanvas"></canvas>
        <b-button v-if="isVideoShown" @click="stopDemo()" variant="secondary" id="stopDemoButton">
          <b-icon icon="stop-fill" ></b-icon>
          Quit
        </b-button>
      </div>
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

// Intervals in ms.
const HAND_ICON_OFF_INTERVAL = 5000;
const HAND_ICON_ON_INTERVAL = 2000;

const stateKind = {
    WELCOME: 'WELCOME', // Welcome screen.
    INIT: 'INIT',       // Loading models, etc.
    WORKING: 'WORKING', // Detecting.
    EXIT: 'EXIT',       // App finished.
    ERROR: 'ERROR',     // Fatal error, cannot work.
}

class Logger {
  constructor(gtag) {
    this.gtag = gtag;
  }

  log(category, action, label, value=1) {
    console.log(category, action, label, value);
    this.gtag.event({
      eventCategory: category,
      eventAction: action,
      eventLabel: label,
      eventValue: value
    });
  }

  logException(action, exception, value=1) {
    console.error(action, exception, value);
    // TODO(ia): restore this
    // this.ga.event({
    //   eventCategory: 'LogError',
    //   eventAction: action,
    //   eventLabel: exception.stack,
    //   eventValue: value
    // });
  }
}

export default {
  data() {
    return {
      state: stateKind.WELCOME,
      isMobile: false,
      progressMessage: 'Loading ...',
      tempResultPicture: null,
      camera: null,
      isVideoReady: false,
      isVideoShown: false,
      isDetecting: false, // Processing a video frame by a DNN, ignore attempts to quit.
      isHandIconOn: false,
      handIconTime: new Date() - HAND_ICON_OFF_INTERVAL,
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
        this.isDetecting = false;
        this.isVideoShown = false;

        const maxInputSize = this.isMobile ? 256 : 512;
        await this.engine.init(maxInputSize, message => {this.progressMessage = message;});
        this.progressMessage = 'Warming up ...';

        const handIcon = new Image();
        let imageLoaded = false;
        handIcon.onload = function() { imageLoaded = true; }
        handIcon.src = 'hand.svg';
        while(!imageLoaded) {
          await sleep(50);
        }        

        while(!this.isVideoReady) {
          await sleep(50);
        }
        console.log('isVideoReady state: ', this.isVideoReady);

        this.state = stateKind.WORKING;

        for(;;) {
          this.isDetecting = false;
          await sleep(50);
          if(!this.isActive) {
            continue;
          }
          if (this.state != stateKind.WORKING) break;

          this.isDetecting = true;

          // Make this in the loop, as if phone screen orientation changes, video will resize.
          const bufferCanvas = document.createElement('canvas');
          bufferCanvas.setAttribute('width', this.camera.videoWidth);
          bufferCanvas.setAttribute('height', this.camera.videoHeight);

          const bufferContext = bufferCanvas.getContext("2d");

          // Flip camera image.
          bufferContext.translate(bufferCanvas.width, 0);
          bufferContext.scale(-1, 1);
          bufferContext.drawImage(this.camera, 0, 0);
          bufferContext.resetTransform();

          const {objects, objectSize} = await this.engine.predict(bufferCanvas);

          // console.log('getting viewCanvas');
          const viewCanvas = document.getElementById('viewCanvas');
          // Make View canvas fit the container and preserve the aspect ratio.
          const viewContainer = document.getElementById('viewContainer');
          const viewRect = viewContainer.getBoundingClientRect();
          const fullWidth = viewRect.width;
          const fullHeight = window.innerHeight - viewRect.y - 2;
          const viewScaleX = fullWidth / bufferCanvas.width;
          const viewScaleY = fullHeight / bufferCanvas.height;
          let viewScale = 1;
          if(viewScaleX < viewScaleY) {
            viewCanvas.setAttribute('width', fullWidth);
            viewCanvas.setAttribute('height', bufferCanvas.height * viewScaleX);
            viewScale = viewScaleX;
          }
          else {
            viewCanvas.setAttribute('height', fullHeight);
            viewCanvas.setAttribute('width', bufferCanvas.width * viewScaleY);
            viewScale = viewScaleY;
          }

          const viewContext = viewCanvas.getContext("2d");
          viewContext.drawImage(bufferCanvas, 0, 0, viewCanvas.width, viewCanvas.height);

          const objScale = viewScale * objectSize / 2;

          const currentTime = new Date();
          if(this.isHandIconOn) {
            if(currentTime - this.handIconTime > HAND_ICON_ON_INTERVAL || objects.length > 0) {
              this.handIconTime = currentTime;
              this.isHandIconOn = false;
            }
            else {
              viewContext.drawImage(handIcon, 
                viewCanvas.width / 2 - objScale, viewCanvas.height / 2 - objScale, 
                2 * objScale, 2 * objScale);

              viewContext.font = '24px arial';
              viewContext.fillStyle = "#0AAF0A";
              viewContext.textAlign = 'center';
              viewContext.textBaseline = 'top';
              viewContext.fillText('show me your hand', 
                viewCanvas.width / 2, 
                viewCanvas.height / 2 + objScale);
            }
          }
          else if(currentTime - this.handIconTime > HAND_ICON_OFF_INTERVAL) {
              this.handIconTime = currentTime;
              this.isHandIconOn = true;
          }

          for(const o of objects) {
            // console.log('object', o);
            const sa = Math.sin(o.angle) * objScale;
            const ca = Math.cos(o.angle) * objScale;
            viewContext.setTransform(ca, sa, -sa, ca, o.x * viewScale, o.y * viewScale);
            viewContext.beginPath();
            viewContext.moveTo(0, 0);
            viewContext.lineTo(1, 0);
            viewContext.moveTo(1, 0);
            viewContext.lineTo(0.8, -0.15);
            viewContext.moveTo(1, 0);
            viewContext.lineTo(0.8, 0.15);
            viewContext.arc(0, 0, 1, 0, 2 * Math.PI);
            viewContext.resetTransform();
            viewContext.lineWidth = 3;
            viewContext.strokeStyle = "#0AAF0A";
            viewContext.stroke();
          }

          this.isVideoShown = true;
        }
      }
      catch(error) {
        this.logger.logException('Images.getPicturesTask.createPictures', error);
        if (this.state != stateKind.WELCOME) this.state = stateKind.ERROR;
      }
      this.isDetecting = false;
    },

    reload() {
      location.reload();
    },

    startDemo() {
      this.state = stateKind.INIT;
      this.startVideo();
      this.getPicturesTask();
    },

    async stopDemo() {
      while(this.state == stateKind.WORKING && this.isDetecting) {
        await sleep(50);
      }
      this.state = stateKind.WELCOME;
    },

    onVideoReady() {
      console.log('Video ready.');
      this.isVideoReady = true;
    },

    async startVideo() {
      if (this.camera !== null) {
        // Camera is already opened by a previous call.
        return;
      }

      this.camera = document.createElement("video");
      if (navigator.mediaDevices.getUserMedia) {
        this.camera.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
      }
      this.camera.onloadeddata = this.onVideoReady;
      this.camera.play();
    },    

    shareUrl() {
      return window.location.href;
    },

    shareTitle() {
      return 'Hand Detector Demo';
    },
  },

  created() {
    // Make globals accessible in Vue rendering code
    this.stateKind = stateKind;
    this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    this.$gtag.event('created', { method: 'Google' }); // TODO(ia): this is a test from a documentation, do we need it?
    this.logger = new Logger(this.$gtag);
    this.logger.log('test category', 'test action', 'test label', 123);
    this.isActive = true;
    this.engine = new Engine(this.logger);
  },

  mounted() {
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

#viewContainer {
  position: absolute;
  left: 0px;
  top: 0px;
  width: 100%;
}

#viewCanvas {
  z-index: -1;
}

#stopDemoButton {
  position: absolute;
  top: 1px;
  left: 1px;
}

.error {
  color: var(--danger);
}

.social-share {
  margin: 0 0 10px 0;
}

/* Limit youtube iframe width on large screen. */
.youtube-super-container {
    max-width: 560px;
}

/* Responsive youtube iframe, see https://www.h3xed.com/web-development/how-to-make-a-responsive-100-width-youtube-iframe-embed */
.youtube-container {
    position: relative;
    width: 100%;
    height: 0;
    padding-bottom: 56.25%;
}

.youtube-video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

.main-text {
  margin: 10px 0 0 0;
}

button {
  margin: 0 0.5rem 0 0;
}

</style>
