#!/bin/bash

# 启动后端服务 (Spring Boot)
(cd backend && mvn spring-boot:run) &

# 启动前端服务 (Vue.js)
(cd frontend && npm run serve) &

# 等待所有子进程结束 (按 Ctrl+C 停止)
wait
