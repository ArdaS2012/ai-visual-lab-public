# Security

This release is designed for local, single-user educational use. Both documented web commands and the optional model runtime bind to loopback. The runtime has no user authentication or multi-user isolation. Do not expose either service through a public tunnel or bind it to a public interface without adding appropriate access controls, resource limits and deployment review.

Use GitHub's **Security → Report a vulnerability** to report sensitive issues privately. Do not put credentials, private prompts or exploit details in a public issue. If private reporting is unavailable, report only that a private contact is needed, without sensitive details.

Install from the lockfiles, review dependency updates, and run `pnpm audit --prod`. Browser inference downloads public models; do not replace model sources with untrusted files. Never upload confidential data to examples or issue attachments.

The web bridge rejects cross-origin experiment POSTs, non-JSON submissions and bodies over 16 KiB. These checks do not replace authentication for a hosted service.
