# API Rate Limits and Errors

Rate limits are per workspace, not per API key. Free: 60 requests/minute.
Pro: 600/minute. Business: 3,000/minute. Enterprise: negotiated.

A 429 response includes a Retry-After header in seconds. Clients should honour
it with exponential backoff and jitter. Retrying immediately makes throttling
worse and can trip the abuse detector.

A 401 means the key is invalid or revoked. A 403 means the key is valid but
lacks scope for that endpoint. These are not retryable.

Bulk operations should use the batch endpoint, which counts as one request
regardless of payload size, up to 500 items.
