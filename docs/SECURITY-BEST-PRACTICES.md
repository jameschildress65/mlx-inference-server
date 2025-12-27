# MLX Server V3 - Security Best Practices

**Version**: 3.0.0-alpha
**Date**: 2025-12-24
**Security Level**: Internal/Private Network Only

---

## Security Model

**V3 Security Posture**: Designed for **trusted internal networks only**

⚠️ **NOT designed for public internet exposure** ⚠️

---

## Core Security Principles

### 1. Network Isolation (CRITICAL)

✅ **DO**:
- Bind to `localhost` (127.0.0.1) only (default)
- Use SSH tunnels for remote access
- Deploy behind VPN
- Use Tailscale/ZeroTier for secure access

❌ **DON'T**:
- Expose to public internet (`0.0.0.0`)
- Use port forwarding without authentication
- Allow untrusted network access

**Configuration**:
```python
# config/server_config.py
config.host = "127.0.0.1"  # Localhost only (SAFE)
# config.host = "0.0.0.0"  # All interfaces (DANGEROUS!)
```

###2. Authentication & Authorization

**Current State**: V3 has **NO authentication** ⚠️

**Rationale**: Designed for single-user local deployment

**For Multi-User**: Add reverse proxy with auth
```bash
# Example: nginx with basic auth
location /v1/ {
    auth_basic "MLX Server";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:11440;
}
```

### 3. Input Validation

**Prompt Injection Risk**: LLMs can be manipulated via prompts

✅ **Mitigations**:
- Max token limits enforced
- Timeout on generation
- Resource limits per request

❌ **Not Protected Against**:
- Malicious prompts (user responsible)
- Jailbreak attempts (model-level issue)

**Recommendation**: Sanitize prompts in your application layer

### 4. Resource Limits

**DoS Protection**:
```python
# Limits enforced:
- max_tokens: 2048 (configurable)
- request_timeout: 120s (configurable)
- idle_timeout: 600s (auto-unload)
```

**Worker Isolation**:
- Worker crashes don't affect orchestrator ✅
- Each model in separate process ✅
- OS-level resource limits apply ✅

---

## Threat Model

### Threats V3 PROTECTS Against

✅ **Worker Crashes**: Process isolation prevents orchestrator crash
✅ **Memory Exhaustion**: Worker auto-unload after idle timeout
✅ **Resource Leaks**: 0 MB memory leaks (process termination)
✅ **Concurrent Abuse**: Request serialization prevents race conditions

### Threats V3 DOES NOT Protect Against

❌ **Unauthorized Access**: No authentication
❌ **Prompt Injection**: LLM-level vulnerability
❌ **Data Exfiltration**: No audit logging
❌ **DDoS**: No rate limiting
❌ **Supply Chain**: Dependencies not verified

---

## Deployment Security

### Local Development

```bash
# Minimal security (localhost only)
config.host = "127.0.0.1"
config.main_port = 11440
config.admin_port = 11441
```

### Internal Network

```bash
# Access via SSH tunnel
ssh -L 11440:localhost:11440 user@remote-machine

# Then access locally
curl http://localhost:11440/v1/completions ...
```

### Production (Internal)

```nginx
# Use reverse proxy with TLS + auth
server {
    listen 443 ssl;
    server_name mlx.internal.company.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    auth_basic "MLX Server";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://localhost:11440;
        proxy_set_header Host $host;
    }
}
```

---

## Firewall Configuration

### macOS Firewall

```bash
# Block external access to ports
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /path/to/python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --block /path/to/python

# Verify
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps
```

### Advanced: pf Firewall

```bash
# /etc/pf.conf
# Block external access to MLX ports
block in proto tcp from any to any port {11440, 11441}
pass in proto tcp from lo0 to any port {11440, 11441}
```

---

## Secrets Management

### API Keys (If Using External Services)

```bash
# Store in environment variables
export OPENAI_API_KEY="..."  # If using for comparison
export HF_TOKEN="..."  # If using private HuggingFace models

# NOT in code
# NOT in config files
# NOT in logs
```

### Model Access

**Private Models**:
```bash
# Use HuggingFace token
huggingface-cli login

# Or environment variable
export HF_TOKEN="hf_..."
```

---

## Monitoring & Auditing

### Log Security

**Current Logging**: Basic request/response logging

**Sensitive Data**: Prompts logged (be careful!)

**Recommendations**:
```python
# Redact sensitive prompts in logs
def sanitize_prompt(prompt):
    if len(prompt) > 100:
        return prompt[:50] + "..." + prompt[-20:]
    return prompt

logger.info(f"Prompt: {sanitize_prompt(prompt)}")
```

### Audit Trail

**Not Implemented**: V3 doesn't track:
- Who made requests
- When requests made
- What data accessed

**Future Enhancement** (Phase 6):
- Request ID tracking
- User attribution
- Audit log export

---

## Dependency Security

### Vulnerability Scanning

```bash
# Check for known vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade mlx-lm transformers
```

### Supply Chain

**Trust Model**:
- MLX: Published by Apple ML Research ✅
- mlx-lm: Official MLX ecosystem ✅
- Transformers: HuggingFace (widely used) ✅

**Verification**:
```bash
# Verify package hashes
pip install --require-hashes ...

# Use private PyPI mirror
pip install --index-url https://pypi.company.com/simple
```

---

## Data Privacy

### Prompt Data

**Storage**: Prompts NOT stored persistently ✅
**Logging**: Prompts logged to `logs/mlx-server-v3.log` ⚠️
**Network**: Prompts never sent externally ✅

**For GDPR/Privacy Compliance**:
```bash
# Disable prompt logging
# Edit logging config to exclude prompts

# Or encrypt logs
# Use encrypted filesystem for logs/
```

### Model Weights

**Storage**: Models cached in `~/.cache/huggingface/`
**Privacy**: Models stay on local machine ✅
**Network**: Downloaded once from HuggingFace, then cached ✅

---

## Incident Response

### Security Incident Checklist

1. **Stop Server**:
   ```bash
   ./bin/mlx-server-v3-daemon.sh stop
   ```

2. **Isolate System**:
   ```bash
   # Disconnect network if compromised
   sudo ifconfig en0 down
   ```

3. **Collect Evidence**:
   ```bash
   cp -r logs/ /secure/backup/
   ps aux > /secure/backup/processes.txt
   ```

4. **Analyze**:
   ```bash
   grep -i "unusual\|suspicious" logs/mlx-server-v3.log
   ```

5. **Remediate**:
   - Patch vulnerabilities
   - Rotate credentials
   - Update dependencies

6. **Resume Service**:
   - After verification only

---

## Security Checklist

### Deployment Security

- [ ] Server bound to localhost only
- [ ] Firewall rules configured
- [ ] Access via SSH tunnel or VPN
- [ ] No public internet exposure
- [ ] TLS if using reverse proxy
- [ ] Authentication if multi-user

### Operational Security

- [ ] Dependencies regularly updated
- [ ] Logs reviewed for anomalies
- [ ] Prompts sanitized (if logging)
- [ ] Secrets in environment vars (not code)
- [ ] Backups encrypted
- [ ] Access controls documented

### Monitoring

- [ ] Health checks automated
- [ ] Resource usage monitored
- [ ] Worker crashes alerted
- [ ] Unusual activity detected

---

## Known Security Issues

### Alpha Version Risks

1. **No Authentication**: Anyone with network access can use server
2. **No Rate Limiting**: Potential DoS via flood requests
3. **No Audit Logging**: Can't track who did what
4. **Prompt Logging**: Sensitive data in logs
5. **No Input Sanitization**: Malicious prompts possible

**Mitigation**: **Deploy in trusted network only**

---

## Future Security Enhancements (Phase 6+)

**Planned**:
- API key authentication
- Rate limiting per client
- Request quota management
- Audit logging
- Metrics export (detect anomalies)
- Input validation framework

**Not Planned**:
- Public internet deployment (out of scope)
- Advanced threat detection
- Encryption at rest

---

## Recommendations by Use Case

### Personal Use (Single User)

✅ Localhost only
✅ No additional security needed
✅ Firewall optional

### Team Use (Trusted Network)

✅ Bind to internal IP or VPN
✅ Reverse proxy with basic auth
✅ TLS certificate
✅ Firewall rules

### NOT RECOMMENDED: Public Internet

❌ V3 NOT designed for public exposure
❌ No authentication
❌ No rate limiting
❌ Potential abuse

**Alternative**: Use managed LLM API services

---

**Document Version**: 1.0
**Last Updated**: 2025-12-24
**Security Contact**: security@example.com
**Classification**: Internal Use Only
