# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab) or by email to the Phexara
security contact. Do not open public issues for security problems.

We acknowledge reports within two working days and aim to provide a remediation plan within
ten working days.

## Supported versions

Security fixes are released for the latest minor version. Deployments should track the
latest release, which CD promotes after the full CI and security pipeline passes.

## Supply chain controls

* Dependencies are pinned in `requirements/*.lock`; the runtime lock is hash verified.
* GitHub Actions are pinned to commit SHAs (`scripts/pin_actions.py`) and kept current by Dependabot.
* Every pull request runs dependency review, pip audit, Bandit, CodeQL, Gitleaks and Trivy.
* Container images are built once, signed keylessly with Sigstore cosign, published with an
  SBOM and SLSA build provenance, and verified before every deployment.
