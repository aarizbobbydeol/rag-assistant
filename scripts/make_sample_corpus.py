"""Generate the fictional demo corpus in ``data/corpus``.

Everything here is invented: Northwind Cartography does not exist, and no text
is copied from any real handbook, policy or API. The point of the corpus is to
be *checkable* - it is dense with numbers, names and deadlines, and several
facts are deliberately split across two documents so the golden set in
``data/golden/qa.jsonl`` can pose genuine multi-hop questions.

The generator is deterministic and idempotent: the document bodies are literal
constants, the PDF is laid out from a fixed page model with a fixed document
id, and a file is only rewritten when its bytes actually change. Re-running the
script therefore never perturbs ``doc_id`` (which is derived from the path) or
the chunk ids downstream.

The golden set records ``relevant_doc_ids`` as the corpus *filenames* written
here, not as ``Document.doc_id`` values: a doc id is the stable hash of the
absolute path, so it differs between machines and checkouts and could never be
committed. That field is only the fallback used when no recorded passage
resolves, and every passage in the golden set resolves, so the readable name is
the more useful thing to store.

Run with the project interpreter::

    .venv/Scripts/python.exe scripts/make_sample_corpus.py
"""

from __future__ import annotations

import argparse
import io
import sys
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"

# --------------------------------------------------------------------------- #
# Markdown documents
# --------------------------------------------------------------------------- #
EMPLOYEE_HANDBOOK = """# Northwind Cartography Employee Handbook

Version 4.2 | Effective 1 April 2025 | Owner: Dana Okafor, VP of People Operations

## Scope and revision history

This handbook applies to all salaried employees of Northwind Cartography, a
mapping-data company founded in 2016 and headquartered in Ann Arbor, Michigan.
As of April 2025 the company employs 214 people across three offices: Ann Arbor,
Lisbon and Wellington. Contractors and agency staff are covered by their own
engagement letters and not by this handbook. The handbook is reviewed once per
fiscal year; the Northwind fiscal year begins on 1 April. Version 4.2 replaces
version 4.1 of 3 April 2024 and changes the carryover cap and the referral bonus.

## Working hours and location

Northwind runs on core hours: every employee is reachable between 10:00 and 16:00
in their own local time, and the remaining hours are theirs to arrange. Meetings
that cross the Ann Arbor and Wellington time zones are scheduled inside the
Wellington morning window of 08:00 to 11:00 NZT.

Employees may work remotely up to three days per week without approval. A fully
remote arrangement requires written approval from a Vice President and is
reviewed every six months. Relocation to a country where Northwind has no legal
entity is not permitted; the company employs staff directly only in the United
States, Portugal and New Zealand.

## Paid time off

Salaried employees accrue 22 days of paid time off per year, credited monthly at
1.83 days on the last calendar day of the month. Up to 10 unused days may be
carried into the next fiscal year, and carried days expire on 31 March if they
are not used. PTO requests of five consecutive days or more should be submitted
at least 14 calendar days in advance through Workbench, the internal HR portal.

Sick leave is separate from PTO and is capped at 10 days per fiscal year. A
doctor's note is required only for an absence of four or more consecutive working
days. Bereavement leave is five days for an immediate family member and two days
otherwise. Northwind observes 11 public holidays per office, published each
December for the following calendar year.

## Probation, reviews and promotion

New employees serve a 90-day probation period, during which either party may end
the employment relationship with one week of notice. After probation the notice
period is four weeks for individual contributors and eight weeks for directors
and above.

Performance reviews run twice a year, in March and September. Ratings are agreed
in a calibration meeting held within 15 business days of the review window
closing. Promotions and salary changes take effect on the first day of the
quarter following calibration. The compensation review budget is set by the
Finance team each January and is not negotiated inside individual reviews.

## Expenses

Expense reports must be filed in Workbench within 10 business days of the
expense being incurred; anything filed later requires a written exception from
the employee's department director. Meals are reimbursed up to 65 US dollars per
day for domestic travel and 95 US dollars per day for international travel. Any
single line item above 500 US dollars requires pre-approval from a department
director, and any item above 5,000 US dollars requires approval from the Chief
Financial Officer, Marguerite Adeyemi.

Reimbursements are paid with the next payroll run following approval. Northwind
pays economy fare for flights under six hours and premium economy for flights of
six hours or more. Rideshare and taxi fares are reimbursed; rental cars require
prior approval because of the company's insurance terms.

## On-call and referrals

Engineers on the production on-call rotation receive a stipend of 45 US dollars
per weekday shift and 70 US dollars per weekend or public-holiday shift, paid
monthly in arrears. The stipend is paid whether or not the engineer is paged.
Rotation mechanics, acknowledgement targets and escalation rules live in the
Incident Response Runbook rather than in this handbook.

The employee referral bonus is 2,500 US dollars, paid in the first payroll run
after the referred hire completes their 90-day probation period. Referrals for
director-level and above roles are not eligible for the bonus.

## Equipment

Every new employee receives a company laptop from IT on their first day, along
with a hardware security key. Laptops are replaced on a four-year cycle or
earlier if repair costs would exceed 40 percent of the replacement price.
Personal devices may be used for email and chat only, and only when enrolled in
mobile device management. Requirements for disk encryption, screen locks and
software installation are set by the Information Security Policy.

## Conduct and grievances

Northwind prohibits harassment, discrimination and retaliation of every kind.
Concerns can be raised with a manager, with People Operations, or anonymously
through the Speak Up line operated by an external provider. People Operations
acknowledges every report within two business days and aims to close an
investigation within 20 business days. Retaliation against a person who raises a
concern in good faith is itself a terminable offence.
"""

SECURITY_POLICY = """# Northwind Cartography Information Security Policy

Version 3.1 | Effective 1 February 2025 | Owner: Priya Raghunathan, Chief Information Security Officer

## Purpose and scope

This policy governs every system that stores or processes Northwind data,
including the Atlas API platform, the internal Workbench portal, and all company
laptops. It applies to employees, contractors and vendors with access to
Northwind systems. The policy is reviewed annually and after any SEV-1 security
incident. Exceptions must be requested in writing and are granted by the Chief
Information Security Officer for a maximum of 90 days.

## Authentication

Passwords must be at least 14 characters long. Northwind does not force periodic
password rotation; passwords are changed only when there is evidence of
compromise, which is the guidance the security team considers most effective in
practice. Every new or changed password is checked against a blocklist of 60,000
previously breached passwords, and reuse of the previous five passwords is
rejected.

Multi-factor authentication is mandatory on every account. Administrators of
production systems must use a FIDO2 hardware security key; time-based one-time
passcodes are acceptable for all other staff. SMS-delivered codes are prohibited
because they are vulnerable to SIM-swap attacks. Idle sessions expire after 30
minutes, and sessions in the production admin console expire after 15 minutes.

## Devices

Company laptops must have full-disk encryption enabled, and IT verifies
encryption within three business days of the device being issued. Screens must
lock automatically after five minutes of inactivity. Software is installed from
the managed catalogue; installing unmanaged software on a machine with
production access is a policy violation. A lost or stolen device must be
reported to the security team within one hour of discovery.

## Access control

Access is granted on the principle of least privilege and is tied to a role, not
to an individual. Managers review the access of their reports every quarter, and
each review must be completed within 15 business days of the quarter ending.
When an employee leaves, all access is revoked within two hours of the
termination taking effect; the offboarding checklist is owned by IT and audited
monthly.

Standing access to production databases is not granted. Engineers request
break-glass access through Workbench, which requires approval from two people
from different teams and expires automatically after four hours. Every
break-glass session is recorded and reviewed the following business day.

## Encryption and secrets

Data at rest is encrypted with AES-256. Data in transit uses TLS 1.3; TLS 1.2 is
the minimum accepted version and older protocol versions are refused at the edge.
Application secrets are stored in the managed secret store and rotated every 180
days. Secrets must never be committed to a repository; the pre-commit scanner
blocks known credential patterns and the security team receives an alert for
every bypass.

## Vulnerability management

Vulnerabilities are remediated on a fixed clock measured from the day the finding
is triaged: critical findings within 7 calendar days, high within 30 days, medium
within 90 days, and low within 180 days. An external penetration test is
commissioned once per year; the most recent test was completed by Halden Security
in November 2024 and produced four findings, all of which were closed by January
2025.

## Awareness

Security awareness training is assigned on the first day of employment and must
be completed within 14 calendar days. Refresher training is annual. Phishing
simulations run once per quarter and the company-wide target is a click rate
below 4 percent; the December 2024 simulation produced a click rate of 3.1
percent.

## Incident classification and data subject requests

Any confirmed or suspected exposure of customer personal data is classified as a
SEV-1 incident regardless of the number of records involved, and the response
timings, paging rules and postmortem deadlines in the Incident Response Runbook
apply from the moment of declaration. The security team is the incident
commander for security incidents.

A customer request to delete personal data is acknowledged within five business
days and completed within 30 calendar days of receipt. Records under a legal
hold are exempt; retention periods and the monthly deletion job are defined in
the Data Retention Standard.
"""

API_REFERENCE = """# Atlas API Reference (v2)

Version 2.8 | Effective 1 March 2025 | Owner: Ivan Petrusek, Staff Engineer, Platform

## Overview

The Atlas API serves Northwind Cartography map, geocoding and routing data over
HTTPS. The base URL is https://api.northwindcarto.example/v2 and every response
is JSON encoded as UTF-8. Requests to the v1 base URL are refused; v1 was retired
on 30 September 2024.

## Authentication

Every request carries an OAuth 2.0 bearer token in the Authorization header.
Access tokens expire after 12 hours and refresh tokens expire after 30 days of
inactivity. Tokens are scoped: maps.read, maps.write, geocode, routes and
usage.read. A request with a valid token but the wrong scope is refused with 403
insufficient_scope, which is the error most often mistaken for an expired token.

## Rate limits

Rate limits are enforced per API key on a sliding one-minute window. The Free
tier allows 60 requests per minute, the Standard tier 600 requests per minute,
and the Enterprise tier 4,000 requests per minute. Every tier tolerates a burst
of twice its limit for up to 10 seconds before shedding load.

A throttled request returns 429 rate_limited with a Retry-After header giving
the number of seconds to wait. Clients should honour Retry-After rather than
retrying immediately; a sustained throttling rate is treated as a production
incident under the thresholds in the Incident Response Runbook.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /maps | List map definitions for the calling account |
| POST | /maps | Create a map definition from a GeoJSON payload |
| GET | /maps/{map_id} | Fetch one map definition |
| GET | /maps/{map_id}/tiles/{z}/{x}/{y} | Fetch a rendered vector tile |
| POST | /geocode | Resolve up to 100 addresses in one call |
| POST | /routes | Compute a route between 2 and 25 waypoints |
| GET | /usage | Return the calling account's usage for a billing period |

## Payload limits

A GeoJSON upload to POST /maps may not exceed 25 megabytes or 10,000 features,
whichever limit is reached first. A payload that parses but describes invalid
geometry is rejected with 422 unprocessable_geometry and a list of the offending
feature indexes. Rendered vector tiles are cached at the edge for 3,600 seconds
and carry an ETag; a conditional request that matches returns 304.

## Pagination

List endpoints use cursor pagination. The page_size parameter defaults to 50 and
may not exceed 200. A response containing more data returns a next_cursor value,
which the client passes back as the cursor parameter. Cursors are opaque and
expire after 15 minutes.

## Idempotency

POST /maps, POST /geocode and POST /routes accept an Idempotency-Key header. The
key is a client-generated string of at most 128 characters, and Atlas retains the
result of a keyed request for 24 hours. Replaying a key with a different request
body returns 409 conflict.

## Error codes

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | invalid_request | The request body or query string is malformed |
| 401 | invalid_token | The bearer token is missing, expired or revoked |
| 403 | insufficient_scope | The token is valid but lacks the required scope |
| 404 | not_found | No resource matches the identifier |
| 409 | conflict | Idempotency key reused with a different body |
| 422 | unprocessable_geometry | The geometry could not be interpreted |
| 429 | rate_limited | The account exceeded its per-minute rate limit |
| 500 | internal_error | Unexpected server fault; safe to retry with backoff |

## Webhooks

Atlas delivers asynchronous job results by webhook. Each delivery is signed with
HMAC-SHA256 and the signature is sent in the X-Atlas-Signature header; the signing
secret is rotated on request and the previous secret stays valid for one hour
after rotation. A delivery that does not receive a 2xx response is retried five
times with exponential backoff over a total window of six hours, after which the
job result is available only by polling.

## Deprecation policy

A breaking change to an endpoint is announced at least 180 days before it takes
effect, and the affected responses carry a Sunset header with the retirement date
during that window. Additive changes, such as a new optional field, are shipped
without notice, so clients must ignore unknown fields.
"""

INCIDENT_RUNBOOK = """# Northwind Cartography Incident Response Runbook

Version 5.0 | Effective 15 January 2025 | Owner: Lucia Marchetti, Director of Reliability

## When to declare

Anyone at Northwind may declare an incident, and declaring one is never held
against the person who declared it. An incident is declared in the Beacon paging
tool, which opens a dedicated chat channel named inc-<number> and a voice bridge.
If you are unsure whether something is an incident, declare it at SEV-3 and let
the incident commander raise or lower the severity.

## Severity levels

Northwind uses three severity levels. There is no SEV-4; anything smaller than a
SEV-3 is filed as a normal defect in the backlog.

SEV-1 is a total outage of the Atlas API, corruption of customer data, or any
confirmed or suspected exposure of customer personal data. The primary on-call
engineer must acknowledge a SEV-1 page within 5 minutes.

SEV-2 is significant degradation short of an outage: elevated latency, a failing
region, or a sustained throttling rate above 2 percent of requests for 10
minutes or more. The acknowledgement target for SEV-2 is 15 minutes.

SEV-3 is a defect affecting a single tenant or a non-critical surface, with a
workaround available. The acknowledgement target for SEV-3 is 4 business hours.

## Paging and escalation

Beacon pages the primary on-call engineer first. If the page is not acknowledged
within 5 minutes, Beacon pages the secondary. If the secondary does not
acknowledge within a further 5 minutes, Beacon escalates to the Director of
Reliability. The on-call rotation is weekly and hands over on Wednesdays at 10:00
Ann Arbor time; the handover note is posted in the rotation channel before the
handover call. Compensation for on-call shifts is set out in the Employee
Handbook.

## Roles

Every SEV-1 and SEV-2 has three named roles held by three different people: the
incident commander, who runs the response and is the only person who changes the
severity; the operations lead, who makes changes to production; and the
communications lead, who owns the status page and customer messaging. For a
security incident the incident commander comes from the security team.

## Communications

For a SEV-1, the communications lead posts an initial status page update within
20 minutes of declaration and refreshes it every 30 minutes until the incident is
resolved. For a SEV-2 the first update is due within 60 minutes and refreshes are
hourly. Customers affected by a confirmed exposure of personal data are notified
within 72 hours of the exposure being confirmed. Internal updates go to the
executive channel at the same cadence as the status page.

## Mitigation practice

Mitigate first and diagnose afterwards. The standard mitigations, in order of
preference, are rolling back the most recent deploy, shifting traffic away from
the failing region, and enabling the read-only mode that serves cached tiles.
Rolling back is expected to complete within 12 minutes of the decision. Any
configuration change made during an incident is recorded in the incident channel
as it happens, because the channel transcript is the primary source for the
timeline.

## Resolution and postmortem

An incident is resolved when customer impact has ended, not when the root cause
is understood. The incident commander declares resolution and posts the closing
summary in the incident channel within one hour.

Postmortems are blameless and are written by the incident commander. A SEV-1
postmortem is due within 5 business days of resolution, and a SEV-2 postmortem is
due within 10 business days. Every postmortem is reviewed in the weekly
reliability forum on Thursdays. Action items are ranked P1 to P3; P1 action items
must be closed within 30 days, and the Director of Reliability reports the open
P1 count to the executive team each month. Postmortem documents themselves are
kept for the period given in the Data Retention Standard.
"""

BENEFITS_SUMMARY = """# Northwind Cartography Benefits Summary (2025)

Version 2025.1 | Effective 1 January 2025 | Owner: Samuel Ortiz, Benefits Manager

## Eligibility

Benefits begin on the first day of employment for salaried employees working 30
hours per week or more. Dependents may be added at hire, during open enrollment,
or within 30 days of a qualifying life event such as marriage, birth or the loss
of other coverage. Open enrollment runs for two weeks each November and the
elections take effect on 1 January.

## Medical, dental and vision

Northwind offers three medical plans in the United States: Trailhead HDHP,
Summit PPO and Compass HMO. Northwind pays 88 percent of the employee premium and
72 percent of dependent premiums on every plan. Employees who choose the
Trailhead HDHP receive an employer health savings account contribution of 1,200
US dollars for individual coverage or 2,400 US dollars for family coverage, paid
in two instalments in January and July.

Dental coverage is provided through Larkspur Dental and vision through Clearline
Vision. Both are included at no employee premium, and both cover preventive care
in full. Employees in Portugal and New Zealand are enrolled in the local private
plans arranged by People Operations rather than in the United States plans.

## Retirement

The Northwind 401(k) plan matches employee contributions dollar for dollar up to
5 percent of base salary. Employees become eligible for the match after
completing the 90-day probation period described in the Employee Handbook, and
the employer match is vested immediately with no cliff. The plan is administered
by Rowan Trust and offers 18 funds including four target-date series.

## Leave

Paid parental leave is 16 weeks for a primary caregiver and 8 weeks for a
secondary caregiver. Leave may be taken in up to three separate blocks and must
be used within 12 months of the birth or placement. Parental leave is paid at
100 percent of base salary and is in addition to the paid time off accrual in the
Employee Handbook.

After five years of continuous service, employees receive a paid sabbatical of
four weeks, which may be combined with paid time off but may not be carried
forward or paid out.

## Stipends and allowances

The wellness stipend is 75 US dollars per month and covers gym membership,
fitness classes, and mental health apps. The stipend is claimed by filing an
expense report in Workbench, so the filing deadline in the Employee Handbook
applies to it as well.

The annual learning budget is 1,500 US dollars per employee for books, courses
and conference tickets. It resets at the start of the fiscal year on 1 April and
does not carry over. New employees receive a one-time home office stipend of 800
US dollars, and the commuter allowance is 110 US dollars per month for employees
who work from an office at least two days per week.

## Insurance and support

Northwind provides life insurance at two times base salary, capped at 500,000 US
dollars, at no cost to the employee. Long-term disability cover replaces 60
percent of base salary after a 90-day elimination period. The employee assistance
programme offers eight counselling sessions per person per year at no charge,
plus unlimited access to the legal and financial helplines.

## How to get help

Benefits questions go to the People Operations queue in Workbench, which
responds within two business days. Claims questions go directly to the carrier;
the carrier contact details are listed on the benefits page of the intranet.
Nothing in this summary changes the terms of the underlying plan documents, and
where the two differ the plan documents govern.
"""

ONBOARDING_GUIDE = """# Northwind Cartography New Hire Onboarding Guide

Version 3.4 | Effective 1 April 2025 | Owner: Rebecca Lindqvist, Onboarding Programme Manager

## Before day one

People Operations sends the welcome packet 10 business days before the start
date. IT ships the laptop and the hardware security key so that they arrive at
least two business days early for remote starters, and holds them at the front
desk for office-based starters. The hiring manager names an onboarding buddy
before the start date; the buddy is an employee from the same team who has been
at Northwind for at least six months.

## Day one

Day one begins at 09:30 local time with a 90-minute orientation session run by
People Operations. New hires collect their laptop and hardware security key from
IT on their first day, enrol in multi-factor authentication, and are assigned the
security awareness training described in the Information Security Policy. The
deadline for completing that training, and the deadline for IT to verify disk
encryption on the newly issued laptop, are both set by that policy rather than by
this guide.

New hires are added to payroll on their start date and receive their first
payslip on the next monthly run, which falls on the 25th of each month or the
preceding business day when the 25th is a weekend or public holiday.

## Week one

The first week is deliberately light on delivery work. The expected milestones
are: a one-to-one with the hiring manager on day one or day two; a walkthrough of
the team's on-call practices; a first pull request merged by the end of the week
for engineering hires; and coffee conversations with three people outside the
immediate team. The onboarding buddy checks in daily during week one and weekly
for the rest of the first month.

## The 30-60-90 checkpoints

Northwind runs three structured checkpoints. At 30 days the new hire and manager
confirm that access, tooling and expectations are in place. At 60 days the new
hire presents a short written summary of one thing they would improve, which is
sent to the team but is never used in performance assessment. At 90 days the
manager confirms completion of the probation period defined in the Employee
Handbook and records the outcome in Workbench.

A new hire who is not meeting expectations at 60 days receives a written support
plan from their manager, reviewed by People Operations before it is shared. The
support plan runs for at least 30 days.

## Systems and access

Accounts are provisioned from the role template attached to the job requisition,
which is why access is requested by role rather than by copying another person's
permissions. A new engineer receives read access to the Atlas API staging
environment on day one; production access is granted only after the on-call
readiness review, which cannot happen before day 30. Every access grant follows
the least-privilege rule and the quarterly review cycle in the Information
Security Policy.

## Onboarding survey and programme metrics

Every new hire receives a survey at day 45. The onboarding programme targets a
satisfaction score of at least 4.2 out of 5, and the 2024 average was 4.4 across
96 respondents. The single most common request in 2024 was earlier access to the
staging environment, which is why staging access moved to day one in version 3.4
of this guide.

## Offboarding, briefly

The mirror image of onboarding is owned by IT and People Operations together.
Managers file the leaver form in Workbench as soon as a departure date is
confirmed. Equipment is returned within five business days of the last working
day, and the revocation of system access follows the timing in the Information
Security Policy rather than the equipment timeline.
"""

MARKDOWN_DOCS: tuple[tuple[str, str], ...] = (
    ("employee-handbook.md", EMPLOYEE_HANDBOOK),
    ("security-policy.md", SECURITY_POLICY),
    ("api-reference.md", API_REFERENCE),
    ("incident-response-runbook.md", INCIDENT_RUNBOOK),
    ("benefits-summary.md", BENEFITS_SUMMARY),
    ("onboarding-guide.md", ONBOARDING_GUIDE),
)

# --------------------------------------------------------------------------- #
# PDF document
# --------------------------------------------------------------------------- #
PDF_FILENAME = "data-retention-standard.pdf"
PDF_TITLE = "Northwind Cartography Data Retention Standard"
PDF_RUNNING_HEAD = "Northwind Cartography - Data Retention Standard v2.0"

# Each page is a sequence of (style, text) blocks; styles are keys of _STYLES.
# Only ASCII is used because the pages are drawn with WinAnsiEncoding Helvetica.
PDF_PAGES: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("title", "Data Retention Standard"),
        ("heading", "Version 2.0 - Effective 1 March 2025"),
        ("heading", "Owner: Tomas Bergqvist, Data Governance Lead"),
        (
            "body",
            "This standard states how long Northwind Cartography keeps each class of "
            "record, who is accountable for the deletion of that class, and how the "
            "deletion is evidenced. It applies to production systems, analytics "
            "warehouses, backups and third-party processors. It replaces version 1.6 "
            "of 12 June 2023, which did not cover the analytics warehouse.",
        ),
        ("heading", "Principles"),
        (
            "bullet",
            "- Collect the minimum. A field that is not collected does not have to be "
            "retained, protected or deleted.",
        ),
        (
            "bullet",
            "- Retain for a stated purpose. Every retention period in this standard "
            "names the purpose that justifies it.",
        ),
        (
            "bullet",
            "- Delete on a schedule, not on request. Deletion is automated so that it "
            "happens whether or not anyone remembers.",
        ),
        (
            "bullet",
            "- Prove it. Each deletion run writes an evidence record that is itself "
            "retained for three years.",
        ),
        (
            "body",
            "Retention periods in this standard are maximums. A team may hold a record "
            "for less time than the maximum, and several do, but no team may hold a "
            "record for longer without an approved legal hold.",
        ),
        ("heading", "Accountability"),
        (
            "body",
            "The Data Governance Lead owns this standard. Each retention class has a "
            "named accountable team, listed in the schedule on page 2. The security "
            "team audits a sample of deletions each quarter and reports the result to "
            "the risk committee, which meets on the second Tuesday of February, May, "
            "August and November.",
        ),
    ),
    (
        ("title", "Retention Schedule"),
        (
            "body",
            "The periods below are measured from the date the record was created, "
            "except for employee records, which are measured from the termination "
            "date. Every period is a maximum and every class is purged by the monthly "
            "job described on page 3.",
        ),
        ("heading", "Customer and product records"),
        (
            "bullet",
            "- Customer support transcripts are retained for 400 days. Accountable "
            "team: Customer Operations.",
        ),
        (
            "bullet",
            "- Atlas API access logs are retained for 90 days. Accountable team: "
            "Platform Engineering.",
        ),
        (
            "bullet",
            "- Rendered map tile caches are retained for 30 days. Accountable team: "
            "Platform Engineering.",
        ),
        (
            "bullet",
            "- Geocoding query text is retained for 14 days and is never joined to an "
            "account identifier. Accountable team: Platform Engineering.",
        ),
        ("heading", "Security, finance and people records"),
        (
            "bullet",
            "- Security audit logs are retained for 545 days, which is 18 months. "
            "Accountable team: Security.",
        ),
        (
            "bullet",
            "- Incident postmortem documents are retained for 5 years. Accountable "
            "team: Reliability.",
        ),
        (
            "bullet",
            "- Billing and invoicing records are retained for 7 years to meet tax law. "
            "Accountable team: Finance.",
        ),
        (
            "bullet",
            "- Job applicant records for candidates who are not hired are retained for "
            "24 months. Accountable team: People Operations.",
        ),
        (
            "bullet",
            "- Employee records are retained for 7 years after the termination date. "
            "Accountable team: People Operations.",
        ),
        ("heading", "Backups"),
        (
            "body",
            "Encrypted database backups are retained for 35 days and are then "
            "destroyed with the rest of the snapshot. Because a backup may still hold "
            "a record that has been deleted from production, a deletion is only "
            "certified as complete once the backup window has also elapsed.",
        ),
    ),
    (
        ("title", "Deletion, Holds and Exceptions"),
        ("heading", "The monthly purge"),
        (
            "body",
            "The automated purge job runs on the fifth day of every month at 02:00 "
            "UTC. It walks each retention class in the schedule, deletes the records "
            "that have passed their maximum period, and writes a signed evidence "
            "record giving the class, the row count and the job identifier. A purge "
            "that fails is retried once; a second failure pages the Platform "
            "Engineering on-call rotation.",
        ),
        ("heading", "Legal holds"),
        (
            "body",
            "A legal hold suspends deletion for the records it names and overrides "
            "every period in this standard. Only the General Counsel may place or "
            "release a hold, and every open hold is reviewed at the quarterly risk "
            "committee meeting. Records under hold are tagged in place rather than "
            "copied, so that a released hold returns the records to their normal "
            "schedule automatically.",
        ),
        ("heading", "Customer deletion requests"),
        (
            "body",
            "A verified customer request to delete personal data is handled inside the "
            "window given in the Information Security Policy and is fulfilled by the "
            "same purge machinery, running out of cycle. The requester receives a "
            "written confirmation naming each retention class that was cleared and "
            "each class that was retained under a legal or tax obligation.",
        ),
        ("heading", "Exceptions"),
        (
            "body",
            "An exception to this standard is requested in Workbench, is approved "
            "jointly by the Data Governance Lead and the Chief Information Security "
            "Officer, and expires after 180 days. Nine exceptions were open at the end "
            "of the 2024 fiscal year and six had been closed by 1 March 2025.",
        ),
    ),
    (
        ("title", "Verification and Reporting"),
        ("heading", "Quarterly sampling"),
        (
            "body",
            "Each quarter the security team draws a sample of 25 records that should "
            "have been deleted and confirms that no copy remains in production, in the "
            "analytics warehouse, or with a third-party processor. A sample with any "
            "surviving record is treated as a control failure and is reported to the "
            "risk committee within 10 business days.",
        ),
        ("heading", "Third-party processors"),
        (
            "body",
            "Processors must contractually delete Northwind data within 60 days of the "
            "end of the contract and must provide a certificate of destruction. The "
            "processor register is reviewed twice a year, in April and October. As of "
            "1 March 2025 Northwind uses 14 processors that handle personal data.",
        ),
        ("heading", "Metrics"),
        (
            "bullet",
            "- Purge success rate, target 100 percent; the 2024 result was 11 "
            "successful runs and 1 retried run.",
        ),
        (
            "bullet",
            "- Sampling pass rate, target 100 percent; the 2024 result was 4 passes "
            "out of 4 quarters.",
        ),
        (
            "bullet",
            "- Median time to fulfil a customer deletion request, target 10 days; the "
            "2024 median was 6 days.",
        ),
        ("heading", "Related documents"),
        (
            "body",
            "This standard should be read with the Information Security Policy, which "
            "sets the deletion request window and the incident classification rules, "
            "and with the Incident Response Runbook, which sets the postmortem "
            "deadlines for the documents retained under the 5-year class.",
        ),
    ),
)

# --------------------------------------------------------------------------- #
# PDF rendering
# --------------------------------------------------------------------------- #
_PAGE_WIDTH = 612.0
_PAGE_HEIGHT = 792.0
_LEFT_X = 72.0
_TOP_Y = 726.0
_BOTTOM_Y = 66.0

# style -> (font size, leading, space after the block, wrap column)
_STYLES: dict[str, tuple[float, float, float, int]] = {
    "title": (17.0, 22.0, 14.0, 58),
    "heading": (12.0, 16.0, 7.0, 76),
    "body": (10.5, 14.5, 11.0, 86),
    "bullet": (10.5, 14.5, 5.0, 84),
}


def _escape_pdf_text(text: str) -> str:
    """Escape the three characters a PDF literal string cannot carry raw."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap(text: str, width: int) -> list[str]:
    # Continuation lines are not indented: a leading space would survive into the
    # extracted text as a separate token boundary and make the golden set's
    # verbatim passages harder to keep honest.
    return textwrap.wrap(text, width=width, subsequent_indent="") or [""]


def _page_operators(blocks: Sequence[tuple[str, str]], page_no: int, page_count: int) -> str:
    """Lay one page out as a PDF content stream.

    Raises ``ValueError`` when the blocks do not fit, so a page model edited to
    be too long fails loudly instead of silently rendering off the paper.
    """
    ops: list[str] = ["BT"]
    y = _TOP_Y
    for style, text in blocks:
        size, leading, space_after, width = _STYLES[style]
        lines = _wrap(text, width)
        ops.append(f"/F1 {size:g} Tf")
        ops.append(f"{leading:g} TL")
        ops.append(f"1 0 0 1 {_LEFT_X:g} {y:g} Tm")
        for index, line in enumerate(lines):
            if index:
                ops.append("T*")
            ops.append(f"({_escape_pdf_text(line)}) Tj")
        y -= leading * len(lines) + space_after
    if y < _BOTTOM_Y:
        raise ValueError(f"page {page_no} overflows: text would end at y={y:.1f}")

    ops.append("/F1 8 Tf")
    ops.append(f"1 0 0 1 {_LEFT_X:g} {_BOTTOM_Y - 18:g} Tm")
    ops.append(f"({_escape_pdf_text(PDF_RUNNING_HEAD)}) Tj")
    ops.append(f"1 0 0 1 {_PAGE_WIDTH - _LEFT_X - 60:g} {_BOTTOM_Y - 18:g} Tm")
    ops.append(f"(Page {page_no} of {page_count}) Tj")
    ops.append("ET")
    return "\n".join(ops)


def build_pdf(pages: Sequence[Sequence[tuple[str, str]]] = PDF_PAGES) -> bytes:
    """Render the retention standard as a real multi-page PDF via pypdf.

    Base-14 Helvetica is used so no font is embedded: the file stays small and
    the bytes stay identical between runs, which is what makes the generator
    idempotent. The pages carry ordinary text operators, so ``pypdf`` extracts
    the text back out and the ingestion layer's page-span logic gets a genuine
    multi-page document to number.
    """
    from pypdf import PageObject, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font_ref = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
    )

    for page_no, blocks in enumerate(pages, start=1):
        page = PageObject.create_blank_page(width=_PAGE_WIDTH, height=_PAGE_HEIGHT)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        writer.add_page(page)
        stream = DecodedStreamObject()
        stream.set_data(_page_operators(blocks, page_no, len(pages)).encode("latin-1"))
        writer.pages[-1].replace_contents(stream)

    writer.add_metadata(
        {
            "/Title": PDF_TITLE,
            "/Author": "Northwind Cartography Records Management",
            "/Subject": "Retention periods, deletion schedule and verification",
            "/Producer": "make_sample_corpus.py",
            "/CreationDate": "D:20250301000000Z",
            "/ModDate": "D:20250301000000Z",
        }
    )
    # A random file id would change the bytes on every run and defeat idempotency.
    writer._ID = None

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def reflow_markdown(text: str) -> str:
    """Join each prose paragraph onto a single line.

    The constants above are hard-wrapped so they stay readable in this file, but
    a hard-wrapped file puts a newline in the middle of every sentence. The
    golden set records ground truth as verbatim passages, and a passage that has
    to reproduce the source's line breaks is a passage nobody can maintain, so
    the written markdown carries one line per paragraph. Headings, table rows
    and list items are structural and are passed through untouched.
    """
    lines: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            lines.append(" ".join(paragraph))
            paragraph.clear()

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "-", "*", "|", ">", "```")):
            flush()
            lines.append(stripped)
        else:
            paragraph.append(stripped)
    flush()
    return "\n".join(lines).strip() + "\n"


def _write_if_changed(path: Path, payload: bytes) -> bool:
    """Write ``payload`` only when it differs. Returns True when written."""
    if path.exists() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return True


def generate_corpus(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[tuple[Path, bool]]:
    """Write every corpus file. Returns ``(path, was_written)`` per document."""
    results: list[tuple[Path, bool]] = []
    for filename, text in MARKDOWN_DOCS:
        target = corpus_dir / filename
        payload = reflow_markdown(text).encode("utf-8")
        results.append((target, _write_if_changed(target, payload)))
    pdf_target = corpus_dir / PDF_FILENAME
    results.append((pdf_target, _write_if_changed(pdf_target, build_pdf())))
    return results


def _word_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if path.suffix == ".pdf":
            continue
        total += len(path.read_text(encoding="utf-8").split())
    return total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory to write the corpus into (default: data/corpus).",
    )
    args = parser.parse_args(argv)

    results = generate_corpus(args.out)
    written = sum(1 for _, changed in results if changed)
    pdf_words = sum(len(text.split()) for page in PDF_PAGES for _, text in page)

    for path, changed in results:
        print(f"{'wrote  ' if changed else 'current'} {path}")
    print(
        f"{len(results)} documents, {written} written, "
        f"{len(results) - written} already current, "
        f"~{_word_count(p for p, _ in results) + pdf_words} words total"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
