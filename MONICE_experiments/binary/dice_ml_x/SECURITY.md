# Security Policy

## Supported Versions

This project is currently under active research and development.  
Security updates are applied only to the latest version on the `main` branch.

| Version        | Supported          |
| -------------- | ------------------ |
| main (latest)  | :white_check_mark: |
| older commits  | :x:                |

> As this is a research codebase, backward compatibility and patching of older versions are not guaranteed.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this repository, please report it responsibly.

### How to report

- Open a **private security advisory** on GitHub (preferred), or  
- Contact [Volkan Bakir](https://www.github.com/creaturerigger) directly via GitHub

Please **do not disclose vulnerabilities publicly** until they are reviewed and addressed.

---

### What to include

To help us understand and fix the issue quickly, include:

- Description of the vulnerability  
- Steps to reproduce  
- Affected components (e.g., DiCE, DP module, utils)  
- Potential impact (e.g., data leakage, model exploitation)
- Suggested fix (optional)

---

### Response timeline

- Initial response: **within 3–5 days**
- Status update: **within 7–10 days**
- Fix (if accepted): depends on severity and complexity

---

### Scope of vulnerabilities

We are especially interested in reports related to:

- **Data leakage risks**
  - Training data exposure
  - Counterfactual reconstruction attacks

- **Differential privacy weaknesses**
  - Improper noise calibration
  - Attacks bypassing DP guarantees

- **Model exploitation**
  - Adversarial attacks on counterfactual generation
  - Membership inference or model inversion

- **Dependency vulnerabilities**
  - Known CVEs in PyTorch, TensorFlow, or other libraries

---

### Out of scope

The following are **not considered security vulnerabilities**:

- Theoretical limitations of explainability methods
- Expected model bias or fairness issues
- Low-impact numerical instability without security implications

---

### Disclosure policy

- Vulnerabilities will be reviewed and, if valid, fixed as soon as possible
- You may be credited for responsible disclosure (unless you prefer anonymity)
- Public disclosure will occur **after a fix is released**

---

### Disclaimer

This repository is primarily intended for **research and experimental use**.
It is **not production-hardened**, and users should exercise caution when deploying it in real-world systems, especially those involving sensitive data.
