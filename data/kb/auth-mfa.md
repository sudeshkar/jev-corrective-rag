# Multi-Factor Authentication

MFA supports TOTP authenticator apps and hardware security keys (WebAuthn). SMS
codes are not supported.

Admins can enforce MFA workspace-wide under Settings > Security. Existing users
are given a 7-day grace period to enrol before being blocked at login.

Recovery codes are shown once at enrolment. If a user loses both their device
and their recovery codes, a workspace admin must reset MFA for that user. There
is no self-service path.

Service accounts and API keys do not use MFA. Rotate API keys instead.
