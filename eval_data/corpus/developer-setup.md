# Developer Setup Guide

This guide takes a new engineer from a fresh laptop to a first merge request. It assumes you have finished the onboarding steps and that IT has given you access to the code repository.

## Prerequisites

You need a macOS or Linux laptop with Python 3.12, uv and Docker Desktop. Give Docker at least 8 GB of memory, or the local stack will stop with out-of-memory errors when the ledger and the queue run together. Windows laptops are not supported for development; use the remote development box instead.

Install the command line tool nwctl from the internal package index. It wraps the common tasks in this guide and checks that your environment matches what CI expects.

## Getting the code

```bash
# clone the monorepo
git clone git@code.northwind.example:platform/monorepo.git
cd monorepo
# install dependencies and git hooks
make bootstrap
```

After bootstrap, run make doctor. It must report 0 errors before you open your first merge request, because CI runs the same checks and will reject the branch. The most common error is a Docker memory limit below 8 GB.

## Local environment

```
# .env.local (never commit this file)
NW_ENV=local
NW_LEDGER_URL=http://localhost:7400
NW_QUEUE_URL=http://localhost:7410
```

The local ledger listens on port 7400 and the queue on port 7410. Secrets for local development come from Vaultkeep; never copy a production secret into your environment file. The file .env.local is listed in .gitignore, and a pre-commit hook refuses commits that contain it.

## Running the tests

```bash
# unit tests, about 4 minutes
make test
# integration tests need the local stack
make stack-up
make test-integration
```

Unit tests run without any services. The integration suite needs the local stack running and takes about 25 minutes on a laptop, so most engineers let CI run it. CI runs the integration suite on every merge request to the main branch, and a red integration run blocks the merge.

## Merge requests

A merge request needs 2 approvals, and one of them must come from a code owner of the files you changed. The title must start with the ticket key, for example PAY-1234, so that the change shows up on the ticket. Merge requests are squash-merged, so write the final commit message in the merge request description.

Keep merge requests small. Reviews are expected within 1 working day; if a review takes longer, ask in the team channel rather than merging with a single approval.

## Deploying to staging

```bash
# deploy your branch to staging
nwctl deploy --env staging --branch my-branch
# follow the logs of the deployment
nwctl logs --env staging --follow
```

Staging deploys are allowed from Monday to Thursday, between 9:00 and 17:00 CET. On Fridays staging is frozen so that the release candidate can be tested. Production is never deployed by hand: changes reach production with the release train, which leaves every Tuesday and Thursday at 11:00 CET.

## Getting help

Ask questions in the #dev-help channel. The platform team runs an on-call rotation that changes every Monday; the on-call engineer answers urgent questions about the local stack and CI within 30 minutes during office hours.
