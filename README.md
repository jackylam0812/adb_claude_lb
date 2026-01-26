# Databricks Claude Load Balancer

一个用于 Claude Code 的负载均衡代理，支持将请求分发到多个 Databricks Claude 端点。

## 功能特性

- 🔄 **负载均衡** - 支持最少请求数 (least_requests)、轮询 (round_robin)、随机 (random) 策略
- 🔌 **多端点支持** - 可配置多个 Databricks workspace 端点
- ⚡ **熔断器** - 自动检测故障端点并临时禁用
- 🔐 **API Key 认证** - 支持自定义 API Key 验证
- 📡 **流式响应** - 完整支持 SSE 流式输出
- 🧠 **Extended Thinking** - 支持 Claude Opus 的思考模式

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn httpx pyyaml
```

### 2. 配置端点

编辑 `config.yaml`：

```yaml
load_balancer:
  strategy: least_requests  # least_requests, round_robin, random
  circuit_breaker_threshold: 5
  circuit_breaker_timeout: 60

auth:
  api_key: your-api-key

endpoints:
  - name: workspace-1
    api_base: https://adb-xxx.azuredatabricks.net/serving-endpoints
    token: dapi_xxx
    weight: 1

  - name: workspace-2
    api_base: https://adb-yyy.azuredatabricks.net/serving-endpoints
    token: dapi_yyy
    weight: 1
```

### 3. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动。

### 4. 配置 Claude Code

设置环境变量：

```bash
export ANTHROPIC_BASE_URL='http://localhost:8000'
export ANTHROPIC_API_KEY='your-api-key'  # 与 config.yaml 中的 api_key 一致
```

## API 端点

| 端点 | 描述 |
|------|------|
| `POST /v1/messages` | 主要的消息 API（兼容 Claude API） |
| `POST /v1/messages/count_tokens` | Token 计数估算 |
| `GET /health` | 健康检查 |
| `GET /stats` | 查看各端点统计信息 |
| `POST /reset` | 重置所有熔断器状态 |

## Docker 部署

```bash
# 构建镜像
docker build -t claude-lb .

# 运行
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml claude-lb
```

## 架构说明

```
Claude Code  ──►  Load Balancer Proxy  ──►  Databricks Workspace 1
                       (localhost:8000)  ──►  Databricks Workspace 2
                                         ──►  Databricks Workspace 3
```

代理使用 Databricks 原生的 Anthropic 端点 (`/anthropic/v1/messages`)，直接透传请求和响应，无需格式转换。

## 支持的模型

| Claude 模型 | Databricks 模型 |
|------------|-----------------|
| claude-sonnet-* | databricks-claude-sonnet-4-5 |
| claude-opus-* | databricks-claude-opus-4-5 |
| claude-haiku-* | databricks-claude-sonnet-4-5 |

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `CONFIG_PATH` | 配置文件路径 | `config.yaml` |

## License

MIT
