const { defineConfig } = require('@vue/cli-service')
module.exports = {
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:8081', // Backend port updated to 8081
        changeOrigin: true
      }
    }
  }
}
