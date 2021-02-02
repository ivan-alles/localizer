// Copyright 2016-2020 Ivan Alles. See also the LICENSE file. 

import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueAnalytics from 'vue-analytics'


Vue.config.productionTip = false

Vue.use(VueAnalytics, {
  id: 'UA-171289527-2',
  router
})

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
