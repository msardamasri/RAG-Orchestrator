# Customer Support Service Levels

Owner: Customer Support. Last reviewed: April 2026. This document describes the support Northwind Payments commits to for its customers. It is the internal reference behind the public service level agreement.

## Support tiers

The support tier of a customer is determined by their contract.

- Standard support: email only, Monday to Friday from 09:00 to 18:00 Central European Time.
- Priority support: email and live chat, 24 hours a day from Monday to Friday.
- Premier support: email, chat and phone, 24 hours a day and 7 days a week, with a named technical account manager.

Every request becomes a ticket with an identifier in the format SUP-######. Customers can open tickets from the dashboard or by email.

## Priority levels

The support engineer assigns a priority when the ticket is opened; the customer may ask for it to be reviewed.

- P1: payment processing is down or failing for most transactions.
- P2: an important feature is degraded, but payments are still being processed.
- P3: a minor problem for which a workaround exists.
- P4: a general question or a feature request.

## Response and resolution targets

Targets are measured from the moment the ticket is created, within the hours covered by the customer's tier.

- P1: first response within 30 minutes and resolution within 4 hours.
- P2: first response within 2 hours and resolution within 12 hours.
- P3: first response within 8 business hours and resolution within 5 business days.
- P4: first response within 2 business days and resolution within 20 business days.

A resolution can be a permanent fix or a workaround accepted by the customer. Time spent waiting for information from the customer does not count.

## Escalation path

If a P1 ticket has no update for 30 minutes, it escalates to the duty manager. If it is still unresolved after 2 hours, the Head of Support takes over communication with the customer. At the 4-hour mark the CTO office is informed and joins the incident call. P2 tickets escalate to the duty manager when they pass 75% of their resolution target. Customers on Premier support may also escalate directly through their technical account manager at any time.

## Status page

Service status is published at status.northwind-payments.example. During a P1 incident the status page is updated at least every 30 minutes, even if there is nothing new to say. Customers can subscribe to updates by email or webhook. The status page is hosted outside our main infrastructure so that it stays online during an outage.

## After a major incident

For every P1 incident the customer receives a root cause report within 3 business days. The report explains what happened, how long it lasted, which transactions were affected and what we are changing to prevent a repeat.

## Uptime commitment and service credits

Northwind commits to 99.95% monthly uptime for the payments API. If monthly uptime falls below 99.95%, the customer receives a credit of 10% of that month's fee. If it falls below 99.5%, the credit is 25% of that month's fee. Credits must be requested within 30 days of the end of the month concerned and are applied to the next invoice; they are never paid in cash.

## Planned maintenance

Planned maintenance takes place on Sundays between 02:00 and 04:00 UTC. Customers are notified at least 5 days in advance, and planned maintenance does not count against the uptime commitment. Emergency maintenance is announced on the status page as early as possible.

## Quality targets

The customer satisfaction target is 92% across all tickets. Each support engineer has 5 randomly chosen tickets reviewed by a peer every month. Tickets reopened by the customer within 7 days count as unresolved for reporting purposes.

## Languages

Support is provided in English, Spanish and Swedish. Tickets written in another language are answered in English, and the first response target is extended by 1 business hour to allow for translation.
