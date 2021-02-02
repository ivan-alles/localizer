<!-- Copyright 2016-2020 Ivan Alles. See also the LICENSE file. -->

<template>
  <b-container>
    <div>
      <h1>Make a Person of Your Dreams</h1>
    </div>
    <template v-if="state === stateKind.WORKING">
      <template v-if="fullPicture === null">
        <div class="flex-container content">
          <div v-for="(picture, index) in pictures" :key="index" class="picture">
            <template v-if="picture.preview !== null">
              <img :src="picture.preview" class="preview-img" @click="showFullPicture(picture)">
              <span v-if="picture.liked">
                <b-icon icon="heart-fill" @click="toggleLike(picture)" class="like-button liked"></b-icon>
              </span>
              <span v-else>
                <b-icon icon="heart" @click="toggleLike(picture)" class="like-button"></b-icon>
              </span>
            </template>
            <template v-else>
              <h4>
                <b-spinner variant="secondary" label="Dreaming"></b-spinner>
                Dreaming
              </h4>
            </template>
          </div>
          <p class="placeholder-picture">
            <template v-if="isScrollingRequiredForNewPictures">
              Scroll for more pictures
            </template>
          </p>
        </div>
        <p class="placeholder-picture"></p>
        <div class="footer">
          <b-container>
            <template v-if="! hasLikes">
              <p>Making random pictures.</p>
              <p>
                <b-icon icon="heart-fill" class="like-in-text"></b-icon>
                Like some pictures to dream similar ones.</p>
            </template>
            <template v-else>
              <div class="flex-container content">
                <b-icon icon="heart-fill" class="like-in-likes"></b-icon>
                <div v-for="(picture, index) in findLikes" :key="index" class="liked-picture">
                  <template v-if="picture.preview !== null">
                    <img :src="picture.preview" class="liked-img" @click="toggleLike(picture)">
                  </template>
                </div>
                <b-button @click="unlikeAll()" variant="secondary">
                  <b-icon icon="trash" ></b-icon>
                </b-button>
              </div>
              <b-row>
                <b-col sm="1">
                  <label>Variance</label>
                </b-col>
                <b-col sm="3" id="variance-slider">
                  <b-form-input v-model="varianceSlider" type="range" min="0" max="4"></b-form-input>
                </b-col>
              </b-row>
            </template>
          </b-container>  
        </div>        
      </template>
      <template v-else>
        <b-button @click="closeFullPicture()" variant="primary">
          <b-icon icon="arrow-left-short" ></b-icon>
          Continue
        </b-button>
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
            description="
            Please put the long URL above to the address bar of your browser."
          >
          <b-button variant="secondary">
            <b-icon icon="envelope" ></b-icon>
          </b-button>
        </ShareNetwork>          
        <template v-if="fullPicture.full !== null && fullPicture.full !== 'ERROR'">
          <img :src="fullPicture.full" class="full-picture">
        </template>
        <template v-else>
          <img :src="fullPicture.preview" class="full-picture">
        </template>
      </template>
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

class Picture {
  // An array of latents.
  latents;

  // Preivew image in data URL format. Other possible values:
  // null: not created yet.
  // 'ERROR': there was an error in generation.
  preview;

  // Full image, values as for the preview.
  full = null;

  // A bool value, true if the picture is liked.
  liked = false;

  creationTime = performance.now();

  constructor(latents, preview, full=null) {
    this.latents = latents;
    this.preview = preview;
    this.full = full;
  }

  /**
   * Lifetime of the cell in ms.
  */
  lifeTime() {
    return performance.now() - this.creationTime;
  }
}

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
      pictures: [],
      varianceSlider: 2,
      fullPicture: null,
      isMobile: false,
      isScrollingRequiredForNewPictures: false,
      progressMessage: 'Loading ...',
      previewScrollTop: null,
    };
  },
  computed: {
    findLikes() {
      let likes = this.pictures.filter(picture => picture.liked);
      return likes;
    },

    hasLikes: function() {
      return this.findLikes.length > 0;
    },
  },  

  methods: {
    /**
    * Generates pictures in the background.
    */
    async getPicturesTask() {
      try {
        await this.engine.init(message => {this.progressMessage = message;});
        this.progressMessage = 'Warming up ...';
        if('show' in this.$route.query) {
          const latents = this.engine.convertURIStringToLatents(this.$route.query['show']);
          const enginePictures = await this.engine.generatePictures([latents], ['preview', 'full']);
          this.checkFatalError(enginePictures);
          const picture = new Picture(enginePictures[0].latents, enginePictures[0].preview, enginePictures[0].full);
          this.pictures.push(picture);
          this.fullPicture = picture;
          this.state = stateKind.WORKING;
        }       

        while(this.state != stateKind.EXIT) {
          await sleep(50);

          if(!this.isActive) {
            continue;
          }

          // Use a local variable, as the member variable can be reset during async calls.
          const fullPicture = this.fullPicture;
          if(fullPicture !== null) {
            if(fullPicture.full === null) {
              const enginePictures = await this.engine.generatePictures([fullPicture.latents], ['full']);
              fullPicture.full = enginePictures[0].full;
            }
            continue;
          }
          
          if(this.isLearningTriggered) {
            this.isLearningTriggered = false;
            
            const likes = this.findLikes;
            const latents = [];
            for(let like of likes) {
                latents.push(like.latents);
            }
            this.engine.learn(latents);
          }

          let picturesInProgress = this.pictures.filter(picture => picture.preview === null && picture.lifeTime() > 100);
          if(picturesInProgress.length > 0) {
            const enginePictures = await this.engine.createPictures(picturesInProgress.length, this.varianceSlider, ['preview']);
            this.checkFatalError(enginePictures);
            // Although the initialization is already done, we postone going to the working state until 
            // the 1st image is created, as it is slower (warm up) and may even fail.
            // This gives the user more time to see the initial screen.
            if(this.state === stateKind.INIT) {
              this.state = stateKind.WORKING;
            }
            for(let i = 0; i < enginePictures.length; ++i) {
              picturesInProgress[i].preview = enginePictures[i].preview;
              picturesInProgress[i].latents = enginePictures[i].latents;
            }
            continue;
          }

          // Are images below the bottom of the screen?
          if(document.documentElement.scrollTop + window.innerHeight < document.documentElement.offsetHeight - 90) {
            this.isScrollingRequiredForNewPictures = true;
            continue;
          }
          this.isScrollingRequiredForNewPictures = false;

          const size = 1;
          let newPictures = [];
          for(let i = 0; i < size; ++i) {
            const picture = new Picture(null, null);
            newPictures.push(picture);
            this.pictures.push(picture);
          }
        }
      }
      catch(error) {
        this.state = stateKind.ERROR;
        this.logger.logException('Images.getPicturesTask.createPictures', error);
        return;
      }
    },

    checkFatalError(enginePictures) {
      for(const enginePicture of enginePictures) {
        if(enginePicture.preview === 'ERROR') {
          // This is usually a fatal error, cannot continue.
          this.state = stateKind.ERROR;
          throw Error('Cannot create preview picture.');
        }
      }
    },

    reload() {
      location.reload();
    },

    toggleLike(picture) {
      picture.liked = !picture.liked;
      this.isLearningTriggered = true;
    },

    showFullPicture(picture) {
      this.previewScrollTop = document.documentElement.scrollTop;
      // Ensure visibility of the title and social share buttons.
      document.documentElement.scrollTop = 0;
      this.fullPicture = picture;
    },

    closeFullPicture() {
      // TODO(ia): restore scroll position so that we see the same area after closing.
      this.fullPicture = null;
    },

    unlikeAll() {
      for(let picture of this.pictures){
        if(picture.liked) {
          picture.liked = false;
          this.isLearningTriggered = true;
        }
      }
    },

    shareUrl() {
      const url = window.location.href + '?show=' + this.engine.convertLatentsToURIString(this.fullPicture.latents);
      return url;
    },

    shareTitle() {
      return 'Make a Person of Your Dreams';
    },
  },

  created() {
    // Make globals accessible in Vue rendering code
    this.stateKind = stateKind;
    this.Picture = Picture;

    this.isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    this.logger = new GoogleAnalyticsLogger(this.$ga);
    this.isActive = true;
    this.isLearningTriggered = false;
    this.engine = new Engine(this.logger);
  },

  mounted() {
    this.getPicturesTask();
  },

  beforeDestroy () {
    this.state = stateKind.EXIT;
  },

  updated() {
    if(this.fullPicture === null && this.previewScrollTop !== null) {
      document.documentElement.scrollTop = this.previewScrollTop;
      this.previewScrollTop = null;
    }
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

.flex-container {
  display: flex;
  flex-wrap: wrap;
}

.picture { 
  width: 180px;
  height: 180px;
  margin: 2px;
  text-align: center;
  /* For like button positioning to work. */
  position: relative;
  border: 1px solid var(--secondary);
  border-radius: 4px;
  box-shadow: 2px 2px 4px #0004;
} 

.picture h4 { 
  margin-top: 5px;
  margin-bottom: 10px;
}

/* A placeholder of the same size as the picture to enable scrolling before generating real pictures. */
.placeholder-picture { 
  width: 180px;
  block-size: 180px;
  margin: 2px;
  border: 1px solid #00000000;
} 

.preview-img {
    height: 100%;
    width: 100%; 
    object-fit: contain;
    cursor: zoom-in;
}

.liked-picture { 
  width: 80px;
  height: 80px;
  margin: 1px;
  text-align: center;
  border: 1px solid var(--secondary);
  border-radius: 2px;
}

.liked-img {
    height: 100%;
    width: 100%; 
    object-fit: contain;
}

.like-in-likes {
  width: 40px;
  height: 80px;
  padding: 5px;
  color: red;
}

.like-button 
{ 
  width: 40px; 
  height: 40px;
  position: absolute;
  bottom: 5px;
  left: 70px;
  width: 100%; 
  color: #f97878;
}

.like-button:hover {
  color: red;
}

.liked {
  color: red;
}

.like-in-text {
  color: red;
}

button {
  margin: 0 0.5rem 0 0;
}

.full-picture {
  border-radius: 4px;
  box-shadow: 2px 2px 4px #0004;
  margin-top: 10px;  
  width: 100%; 
  height: 100%; 
}

.error {
  color: var(--danger);
}

.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  padding: 2px 5px 2px 5px;
  box-shadow: 0 -5px 5px -5px #0004;
  /* TODO(ia): add shadow at the top.*/
  background-color: #f7f7f9f0;
}

</style>
