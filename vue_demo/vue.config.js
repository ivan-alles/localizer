module.exports = {
  publicPath: process.env.NODE_ENV === 'production' ? 
    '/localizer' : '/',
  configureWebpack: {
    devtool: 'source-map'
  }
}