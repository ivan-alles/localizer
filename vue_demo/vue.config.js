module.exports = {
  publicPath: process.env.NODE_ENV === 'production' ? 
    '/preference-model' : '/',
  /* devServer: {
      proxy: 'https://github.com/ivan-alles/preference-model/releases/download/0.1/',
  }*/
  configureWebpack: {
    devtool: 'source-map'
  }
}