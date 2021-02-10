// Copyright 2021 Ivan Alles. See also the LICENSE file. 

import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueGtag from 'vue-gtag'


Vue.config.productionTip = false

Vue.use(VueGtag, {
    config: { id: 'G-VMZBN6EQSD' }
})

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
