<template>
  <div class="sidebar">
    <div class="logo">
      <h2>DerainSys</h2>
    </div>
    <ul class="nav-links">
      <li>
        <router-link to="/" active-class="active">
          <i class="el-icon-magic-stick"></i> AI 去雨
        </router-link>
      </li>
      <li>
        <router-link to="/stitch" active-class="active">
          <i class="el-icon-picture-outline-round"></i> 图片拼接
        </router-link>
      </li>
      <li>
        <router-link to="/crop" active-class="active">
          <i class="el-icon-crop"></i> 智能裁剪
        </router-link>
      </li>
      <li>
        <router-link to="/color" active-class="active">
          <i class="el-icon-view"></i> 色彩调整
        </router-link>
      </li>
    </ul>
    <div class="footer">
      V 2.0
    </div>
  </div>
</template>

<style scoped>
.sidebar {
  width: 240px;
  height: 100vh;
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  padding: 30px 20px;
  box-sizing: border-box;
}

.logo h2 {
  color: #fff;
  font-size: 24px;
  margin-bottom: 40px;
  text-align: center;
  background: linear-gradient(to right, #6EE7B7, #3B82F6);
  -webkit-background-clip: text;
  color: transparent;
  font-weight: bold;
}

.nav-links {
  list-style: none;
  padding: 0;
  flex: 1;
}

.nav-links li {
  margin-bottom: 12px;
}

.nav-links a {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  text-decoration: none;
  color: rgba(255, 255, 255, 0.7);
  border-radius: 12px;
  transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
  font-size: 16px;
  position: relative;
  overflow: hidden;
}

.nav-links a:hover {
    background: rgba(255, 255, 255, 0.15);
    color: #fff;
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.nav-links a.active {
  background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
  color: #fff;
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5);
  transform: scale(1.02);
}

.nav-links i {
  margin-right: 10px;
}

.footer {
  text-align: center;
  color: rgba(255, 255, 255, 0.2);
  font-size: 12px;
}
</style>
