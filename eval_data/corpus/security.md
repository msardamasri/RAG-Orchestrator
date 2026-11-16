# Information Security Policy

Owner: Security team. Last reviewed: April 2026. Northwind Payments moves money for its customers, so every employee is part of our defence. This policy sets the minimum rules; teams handling card data follow additional controls.

## Passwords

- Passwords must be at least 14 characters long.
- Every password is generated and stored in Vaultkeep, the company password manager.
- You do not need to change a password on a schedule; change it only when a compromise is suspected.
- A new password may not match any of your last 10 passwords.

Never share a password over chat or email. Shared service credentials live in a team vault with an owner who reviews access every quarter.

## Multi-factor authentication

MFA is mandatory on every company account, without exceptions. The accepted methods are the authenticator app and hardware security keys. One-time codes sent by SMS are not accepted as a second factor. Engineers with production access must use a hardware security key, and they receive two keys so that one can be kept as a backup.

## Devices

All laptops and phones that access company data must use full-disk encryption. The screen must lock automatically after 5 minutes of inactivity. Operating system updates are installed within 7 days of release, and patches rated critical within 48 hours. On public or hotel Wi-Fi, connect through the company VPN before opening any internal tool. Do not install browser extensions that are not on the approved list.

## Reporting an incident

Report any suspected security incident within 1 hour of noticing it. Write to security@northwind-payments.example or post in the #sec-incidents channel; outside office hours the on-call security engineer is paged automatically from that channel. Reporting quickly matters more than being sure: nobody is penalised for a false alarm or for reporting their own mistake.

## Severity levels

The security engineer on call assigns a severity level to each report.

- SEV-1: confirmed exposure of customer data or loss of control over a production system. Response starts within 15 minutes and the executive team is informed.
- SEV-2: a compromise contained to one employee account or one internal system. Response starts within 4 hours.
- SEV-3: a policy violation or suspicious activity with no confirmed impact. Response starts within 2 business days.

A written post-incident review is published within 5 business days for every SEV-1 and SEV-2 incident. Reviews are blameless and focus on what to change.

## Phishing

If an email looks suspicious, use the "Report Phish" button in the mail client instead of forwarding it. If you already clicked a link or entered your credentials, report it as an incident straight away. The Security team runs a simulated phishing exercise once per quarter; results are shared per department, never per person.

## Data classification

Information is labelled Public, Internal, Confidential or Restricted. Restricted data, which includes card numbers and customer identity documents, may never be copied to removable media or to personal cloud storage. Confidential documents may be shared outside the company only under a signed non-disclosure agreement.

## Training

Every employee completes the security awareness course once a year. The course takes about 45 minutes and ends with a short quiz; the pass mark is 80%. Access to production systems is suspended for anyone whose course is more than 30 days overdue.

## Visitors and offices

Visitors sign in at reception, wear a visible badge and are accompanied at all times. Do not hold the door for people you do not recognise. Lock your screen whenever you leave your desk, and use the shredding bins for printed documents.

## Access reviews

Managers review the access rights of their team every 6 months. When someone changes role, access from the previous role is removed within 5 working days. When someone leaves, all accounts are disabled on the last working day by 18:00 local time.
