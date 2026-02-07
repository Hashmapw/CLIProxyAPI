## Codex Session ID Hack机制


### Session ID 缓存机制：                                                                                                                                       
  - prompt_cache_key、Conversation_id、Session_id 三者使用相同的缓存 ID，在请求间复用以保持会话连续性                                                   
  - 缓存按 {model}-{user_id} 隔离，不同用户使用独立的 session
  - 缓存有效期 3 小时，过期后自动重新生成
  - 线程安全实现，后台每 15 分钟清理过期条目

### 错误自动恢复：
  - HTTP 错误响应（4xx、5xx）时自动清除缓存，下次请求重新生成 session ID
  - 流式数据中检测 type: "error"（如 permission_error、rate_limit_error），提取错误码和消息，清除缓存并终止流
  - Scanner 读取错误时同样清除缓存
  - 覆盖 Execute（非流式）、ExecuteStream（流式）、executeCompact（紧凑模式）三种请求路径