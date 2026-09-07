# Northwind Cartography Information Security Policy

Version 3.1 | Effective 1 February 2025 | Owner: Priya Raghunathan, Chief Information Security Officer

## Purpose and scope

This policy governs every system that stores or processes Northwind data, including the Atlas API platform, the internal Workbench portal, and all company laptops. It applies to employees, contractors and vendors with access to Northwind systems. The policy is reviewed annually and after any SEV-1 security incident. Exceptions must be requested in writing and are granted by the Chief Information Security Officer for a maximum of 90 days.

## Authentication

Passwords must be at least 14 characters long. Northwind does not force periodic password rotation; passwords are changed only when there is evidence of compromise, which is the guidance the security team considers most effective in practice. Every new or changed password is checked against a blocklist of 60,000 previously breached passwords, and reuse of the previous five passwords is rejected.

Multi-factor authentication is mandatory on every account. Administrators of production systems must use a FIDO2 hardware security key; time-based one-time passcodes are acceptable for all other staff. SMS-delivered codes are prohibited because they are vulnerable to SIM-swap attacks. Idle sessions expire after 30 minutes, and sessions in the production admin console expire after 15 minutes.

## Devices

Company laptops must have full-disk encryption enabled, and IT verifies encryption within three business days of the device being issued. Screens must lock automatically after five minutes of inactivity. Software is installed from the managed catalogue; installing unmanaged software on a machine with production access is a policy violation. A lost or stolen device must be reported to the security team within one hour of discovery.

## Access control

Access is granted on the principle of least privilege and is tied to a role, not to an individual. Managers review the access of their reports every quarter, and each review must be completed within 15 business days of the quarter ending. When an employee leaves, all access is revoked within two hours of the termination taking effect; the offboarding checklist is owned by IT and audited monthly.

Standing access to production databases is not granted. Engineers request break-glass access through Workbench, which requires approval from two people from different teams and expires automatically after four hours. Every break-glass session is recorded and reviewed the following business day.

## Encryption and secrets

Data at rest is encrypted with AES-256. Data in transit uses TLS 1.3; TLS 1.2 is the minimum accepted version and older protocol versions are refused at the edge. Application secrets are stored in the managed secret store and rotated every 180 days. Secrets must never be committed to a repository; the pre-commit scanner blocks known credential patterns and the security team receives an alert for every bypass.

## Vulnerability management

Vulnerabilities are remediated on a fixed clock measured from the day the finding is triaged: critical findings within 7 calendar days, high within 30 days, medium within 90 days, and low within 180 days. An external penetration test is commissioned once per year; the most recent test was completed by Halden Security in November 2024 and produced four findings, all of which were closed by January 2025.

## Awareness

Security awareness training is assigned on the first day of employment and must be completed within 14 calendar days. Refresher training is annual. Phishing simulations run once per quarter and the company-wide target is a click rate below 4 percent; the December 2024 simulation produced a click rate of 3.1 percent.

## Incident classification and data subject requests

Any confirmed or suspected exposure of customer personal data is classified as a SEV-1 incident regardless of the number of records involved, and the response timings, paging rules and postmortem deadlines in the Incident Response Runbook apply from the moment of declaration. The security team is the incident commander for security incidents.

A customer request to delete personal data is acknowledged within five business days and completed within 30 calendar days of receipt. Records under a legal hold are exempt; retention periods and the monthly deletion job are defined in the Data Retention Standard.
