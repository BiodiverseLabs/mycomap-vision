# AWS setup (one time, done by a person)

Photos live in a private S3 bucket, `YOUR-BUCKET`, in **us-east-2** (the
region of the MycoMap Lightsail box). A small EC2 instance downloads them straight
into the bucket and shuts itself down when done.

Claude never sees AWS keys. You create a limited user, and enter its key yourself.

## 1. Install the AWS CLI (this PC, PowerShell)

```
winget install Amazon.AWSCLI
```

## 2. Instance role (AWS console → IAM → Roles → Create role)

- Trusted entity: **AWS service**, use case **EC2**.
- Skip the managed policies. Name it exactly `mycomap-vision-instance`.
- Open the role → **Add permissions → Create inline policy → JSON**, paste
  [`instance-policy.json`](instance-policy.json), name it `vision-bucket`.

It can read and write the vision bucket and nothing else, and cannot delete.

## 3. Ops user (IAM → Users → Create user)

- Name `mycomap-vision-ops`, no console access.
- **Attach policies directly → Create policy → JSON**, paste
  [`ops-policy.json`](ops-policy.json), name it `mycomap-vision-ops`, attach it.
- Open the user → **Security credentials → Create access key → Command Line
  Interface**. Keep the window open for step 4.

It can create and use only the vision bucket (not delete it or make it public),
launch instances only when tagged `Project=mycomap-vision`, stop or terminate
only those, and hand them only the role above.

## 4. Enter the key (this PC, PowerShell, typed by you)

```
aws configure --profile mycomap-vision
```

Region `us-east-2`, output `json`. Check it with:

```
aws sts get-caller-identity --profile mycomap-vision
```

## 5. Optional: budget alarm (Billing → Budgets → Create budget)

A monthly cost budget, e.g. $25, emailing you at 80%.

## Running the downloader

```
.venv/Scripts/mv aws-launch-downloader            # large photos, stops after 120 h at most
```

It creates the bucket (private, encrypted) if needed, uploads the committed code
and a snapshot of the manifest, and starts a `t3.small`. The instance's log is
copied to `s3://YOUR-BUCKET/runs/<run>/download.log` every 15 minutes, and
the manifest to `s3://YOUR-BUCKET/manifest/manifest.sqlite` every 10.

Don't write to the local manifest while it runs. When the instance has gone
(terminated), bring its manifest home:

```
.venv/Scripts/mv aws-pull-manifest
```
