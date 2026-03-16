/**
 * Cursor2API v2 - 入口
 *
 * 将 Cursor 文档页免费 AI 接口代理为 Anthropic Messages API
 * 通过提示词注入让 Claude Code 拥有完整工具调用能力
 */

import 'dotenv/config';
import { createRequire } from 'module';
import express from 'express';
import { getConfig } from './config.js';
import { handleMessages, listModels, countTokens } from './handler.js';
import { handleOpenAIChatCompletions, handleOpenAIResponses } from './openai-handler.js';
import { serveLogViewer, apiGetLogs, apiGetRequests, apiGetStats, apiGetPayload, apiLogsStream } from './log-viewer.js';

// 从 package.json 读取版本号，统一来源，避免多处硬编码
const require = createRequire(import.meta.url);
const { version: VERSION } = require('../package.json') as { version: string };


const app = express();
const config = getConfig();

// 解析 JSON body（增大限制以支持 base64 图片，单张图片可达 10MB+）
app.use(express.json({ limit: '50mb' }));

// CORS
app.use((_req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', '*');
    if (_req.method === 'OPTIONS') {
        res.sendStatus(200);
        return;
    }
    next();
});

// ★ 日志查看器（在鉴权之前，始终可访问）
app.get('/logs', serveLogViewer);
app.get('/api/logs', apiGetLogs);
app.get('/api/requests', apiGetRequests);
app.get('/api/stats', apiGetStats);
app.get('/api/payload/:requestId', apiGetPayload);
app.get('/api/logs/stream', apiLogsStream);

// ★ API 鉴权中间件：配置了 authTokens 则需要 Bearer token
app.use((req, res, next) => {
    // 跳过无需鉴权的路径
    if (req.method === 'GET' || req.path === '/health') {
        return next();
    }
    const tokens = config.authTokens;
    if (!tokens || tokens.length === 0) {
        return next(); // 未配置 token 则全部放行
    }
    const authHeader = req.headers['authorization'] || req.headers['x-api-key'];
    if (!authHeader) {
        res.status(401).json({ error: { message: 'Missing authentication token. Use Authorization: Bearer <token>', type: 'auth_error' } });
        return;
    }
    const token = String(authHeader).replace(/^Bearer\s+/i, '').trim();
    if (!tokens.includes(token)) {
        console.log(`[Auth] 拒绝无效 token: ${token.substring(0, 8)}...`);
        res.status(403).json({ error: { message: 'Invalid authentication token', type: 'auth_error' } });
        return;
    }
    next();
});

// ==================== 路由 ====================

// Anthropic Messages API
app.post('/v1/messages', handleMessages);
app.post('/messages', handleMessages);

// OpenAI Chat Completions API（兼容）
app.post('/v1/chat/completions', handleOpenAIChatCompletions);
app.post('/chat/completions', handleOpenAIChatCompletions);

// OpenAI Responses API（Cursor IDE Agent 模式）
app.post('/v1/responses', handleOpenAIResponses);
app.post('/responses', handleOpenAIResponses);

// Token 计数
app.post('/v1/messages/count_tokens', countTokens);
app.post('/messages/count_tokens', countTokens);

// OpenAI 兼容模型列表
app.get('/v1/models', listModels);

// 健康检查
app.get('/health', (_req, res) => {
    res.json({ status: 'ok', version: VERSION });
});

// 根路径
app.get('/', (_req, res) => {
    res.json({
        name: 'cursor2api',
        version: VERSION,
        description: 'Cursor Docs AI → Anthropic & OpenAI & Cursor IDE API Proxy',
        endpoints: {
            anthropic_messages: 'POST /v1/messages',
            openai_chat: 'POST /v1/chat/completions',
            openai_responses: 'POST /v1/responses',
            models: 'GET /v1/models',
            health: 'GET /health',
            log_viewer: 'GET /logs',
        },
        usage: {
            claude_code: 'export ANTHROPIC_BASE_URL=http://localhost:' + config.port,
            openai_compatible: 'OPENAI_BASE_URL=http://localhost:' + config.port + '/v1',
            cursor_ide: 'OPENAI_BASE_URL=http://localhost:' + config.port + '/v1 (选用 Claude 模型)',
        },
    });
});

// ==================== 启动 ====================

app.listen(config.port, () => {
    const auth = config.authTokens?.length ? `${config.authTokens.length} token(s)` : 'open';
    console.log('');
    console.log(`  \x1b[36m⚡ Cursor2API v${VERSION}\x1b[0m`);
    console.log(`  ├─ Server:  \x1b[32mhttp://localhost:${config.port}\x1b[0m`);
    console.log(`  ├─ Model:   ${config.cursorModel}`);
    console.log(`  ├─ Auth:    ${auth}`);
    console.log(`  └─ Logs:    \x1b[35mhttp://localhost:${config.port}/logs\x1b[0m`);
    console.log('');
});
