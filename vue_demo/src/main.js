// Copyright 2021 Ivan Alles. See also the LICENSE file. 

import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueAnalytics from 'vue-analytics'


Vue.config.productionTip = false

Vue.use(VueAnalytics, {
  id: 'G-VMZBN6EQSD',
  router
})

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
