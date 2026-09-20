# SSO and SAML Configuration

SAML 2.0 single sign-on is available on Business and Enterprise plans. Configure
it under Settings > Security > SSO. You will need your IdP metadata URL and a
verified domain.

Domain verification requires adding a TXT record to your DNS. Propagation takes
up to 24 hours. SSO cannot be enforced until at least one domain is verified.

SCIM user provisioning is Enterprise-only. It syncs every 40 minutes and
deprovisions users within one sync cycle of removal from the IdP group.

If SSO is misconfigured and locks everyone out, the workspace owner can use the
break-glass recovery link emailed at setup time. It bypasses SSO for 60 minutes.
