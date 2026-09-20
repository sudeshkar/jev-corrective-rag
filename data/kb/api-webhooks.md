# Webhooks

Webhook endpoints must respond with a 2xx within 5 seconds. Slower responses are
treated as failures.

Failed deliveries retry 6 times with exponential backoff over roughly 24 hours.
After that the event is dropped and the endpoint is marked unhealthy.

Every payload is signed with HMAC-SHA256 using your signing secret, in the
X-Signature header. Always verify the signature before trusting the payload.
Compare using a constant-time comparison.

Webhook delivery is at-least-once. Your handler must be idempotent; deduplicate
on the event id.
