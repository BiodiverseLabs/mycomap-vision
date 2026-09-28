# AWS setup (one time, done by a person)

Photos live in a private S3 bucket. A small EC2 instance downloads them straight
into the bucket and shuts itself down when done.

Nobody's keys go in the repo or reach an AI assistant: you create a limited user
and enter its key yourself.

## 0. Settings

Put your bucket name, region and AWS profile name in `.env` (see
`.env.example`): `MV_S3_BUCKET`, `MV_AWS_REGION`, `MV_AWS_PROFILE`, and optionally
`MV_INSTANCE_ROLE` (default `mycomap-vision-instance`). Then print the two IAM
policies, filled in for them:

```
.venv/Scripts/mv aws-policies
```

The templates are [`instance-policy.template.json`](instance-policy.template.json)
and [`ops-policy.template.json`](ops-policy.template.json).

## 1. Install the AWS CLI (PowerShell)

```
winget install Amazon.AWSCLI
```

## 2. Instance role (AWS console → IAM → Roles → Create role)

- Trusted entity: **AWS service**, use case **EC2**.
- Skip the managed policies. Name it exactly as `MV_INSTANCE_ROLE`.
- Open the role → **Add permissions → Create inline policy → JSON**, paste the
  printed `instance-policy.json`, name it `vision-bucket`.

It can read and write the vision bucket and nothing else, and cannot delete.

## 3. Ops user (IAM → Users → Create user)

- A name such as `mycomap-vision-ops`, no console access.
- **Attach policies directly → Create policy → JSON**, paste the printed
  `ops-policy.json`, name it, attach it.
- Open the user → **Security credentials → Create access key → Command Line
  Interface**. Keep the window open for step 4.

It can create and use only the vision bucket (not delete it or make it public),
launch instances only when tagged `Project=mycomap-vision`, stop or terminate
only those, and hand them only the role above.

## 4. Enter the key (PowerShell, typed by you)

```
aws configure --profile <MV_AWS_PROFILE>
```

Use your `MV_AWS_REGION`, output `json`. Check it with:

```
aws sts get-caller-identity --profile <MV_AWS_PROFILE>
```

## 5. Optional: budget alarm (Billing → Budgets → Create budget)

A monthly cost budget, e.g. $25, emailing you at 80%.

## Running the downloader

```
.venv/Scripts/mv aws-launch-downloader            # large photos, stops after 120 h at most
```

It creates the bucket (private, encrypted) if needed, uploads the committed code
and a snapshot of the manifest, and starts a `t3.small`. The instance's log is
copied to `s3://<bucket>/runs/<run>/download.log` every 15 minutes, and the
manifest to `s3://<bucket>/manifest/manifest.sqlite` every 10.

Don't write to the local manifest while it runs. When the instance has gone
(terminated), bring its manifest home:

```
.venv/Scripts/mv aws-pull-manifest
```
