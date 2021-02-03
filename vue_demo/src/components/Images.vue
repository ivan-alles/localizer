<!-- Copyright 2016-2020 Ivan Alles. See also the LICENSE file. -->

<template>
  <b-container>
    <div>
      <h1>Localizer</h1>
    </div>
    <template v-if="state === stateKind.WORKING">
      <img :src="tempResultPicture" class="full-picture">
      <img src="test_image.png" class="full-picture" id="cameraImage">
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

// import { Engine } from '@/server-engine'
import { Engine } from '@/client-engine'

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

        this.state = stateKind.WORKING;

      await sleep(1000);
        while(this.state != stateKind.EXIT) {
          await sleep(50);
          if(!this.isActive) {
            continue;
          }
          //this.tempResultPicture = await this.engine.predict(document.getElementById('videoElement'));
          this.tempResultPicture = await this.engine.predict(this.camera);
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
    async startVideo() {
      // this.camera = document.querySelector("#videoElement");
      this.camera = document.createElement("video");
      console.log(this.camera);
      if (navigator.mediaDevices.getUserMedia) {
        this.camera.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
      }
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


.full-picture {
  border-radius: 4px;
  box-shadow: 2px 2px 4px #0004;
  margin-top: 10px;  
  width: 528px; 
  height: 400px; 
}

.error {
  color: var(--danger);
}

#videoElement {
	width: 528px;
	height: 400px;
	background-color: #666;
}

</style>
