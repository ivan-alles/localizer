// Copyright 2021 Ivan Alles. See also the LICENSE file.

import Vue from 'vue'
import VueRouter from 'vue-router'
import { BootstrapVue, IconsPlugin, FormInputPlugin } from 'bootstrap-vue'
import VueSocialSharing from 'vue-social-sharing'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { faVk, faFacebook, faTwitter } from '@fortawesome/free-brands-svg-icons'

import Home from '../views/Home.vue'

import '../custom.scss'

library.add(faVk)
library.add(faFacebook)
library.add(faTwitter)
Vue.component('font-awesome-icon', FontAwesomeIcon)

Vue.use(BootstrapVue)
Vue.use(IconsPlugin)
Vue.use(FormInputPlugin)
Vue.use(VueRouter)
Vue.use(VueSocialSharing)

  const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
