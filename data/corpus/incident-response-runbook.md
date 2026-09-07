# Northwind Cartography Incident Response Runbook

Version 5.0 | Effective 15 January 2025 | Owner: Lucia Marchetti, Director of Reliability

## When to declare

Anyone at Northwind may declare an incident, and declaring one is never held against the person who declared it. An incident is declared in the Beacon paging tool, which opens a dedicated chat channel named inc-<number> and a voice bridge. If you are unsure whether something is an incident, declare it at SEV-3 and let the incident commander raise or lower the severity.

## Severity levels

Northwind uses three severity levels. There is no SEV-4; anything smaller than a SEV-3 is filed as a normal defect in the backlog.

SEV-1 is a total outage of the Atlas API, corruption of customer data, or any confirmed or suspected exposure of customer personal data. The primary on-call engineer must acknowledge a SEV-1 page within 5 minutes.

SEV-2 is significant degradation short of an outage: elevated latency, a failing region, or a sustained throttling rate above 2 percent of requests for 10 minutes or more. The acknowledgement target for SEV-2 is 15 minutes.

SEV-3 is a defect affecting a single tenant or a non-critical surface, with a workaround available. The acknowledgement target for SEV-3 is 4 business hours.

## Paging and escalation

Beacon pages the primary on-call engineer first. If the page is not acknowledged within 5 minutes, Beacon pages the secondary. If the secondary does not acknowledge within a further 5 minutes, Beacon escalates to the Director of Reliability. The on-call rotation is weekly and hands over on Wednesdays at 10:00 Ann Arbor time; the handover note is posted in the rotation channel before the handover call. Compensation for on-call shifts is set out in the Employee Handbook.

## Roles

Every SEV-1 and SEV-2 has three named roles held by three different people: the incident commander, who runs the response and is the only person who changes the severity; the operations lead, who makes changes to production; and the communications lead, who owns the status page and customer messaging. For a security incident the incident commander comes from the security team.

## Communications

For a SEV-1, the communications lead posts an initial status page update within 20 minutes of declaration and refreshes it every 30 minutes until the incident is resolved. For a SEV-2 the first update is due within 60 minutes and refreshes are hourly. Customers affected by a confirmed exposure of personal data are notified within 72 hours of the exposure being confirmed. Internal updates go to the executive channel at the same cadence as the status page.

## Mitigation practice

Mitigate first and diagnose afterwards. The standard mitigations, in order of preference, are rolling back the most recent deploy, shifting traffic away from the failing region, and enabling the read-only mode that serves cached tiles. Rolling back is expected to complete within 12 minutes of the decision. Any configuration change made during an incident is recorded in the incident channel as it happens, because the channel transcript is the primary source for the timeline.

## Resolution and postmortem

An incident is resolved when customer impact has ended, not when the root cause is understood. The incident commander declares resolution and posts the closing summary in the incident channel within one hour.

Postmortems are blameless and are written by the incident commander. A SEV-1 postmortem is due within 5 business days of resolution, and a SEV-2 postmortem is due within 10 business days. Every postmortem is reviewed in the weekly reliability forum on Thursdays. Action items are ranked P1 to P3; P1 action items must be closed within 30 days, and the Director of Reliability reports the open P1 count to the executive team each month. Postmortem documents themselves are kept for the period given in the Data Retention Standard.
