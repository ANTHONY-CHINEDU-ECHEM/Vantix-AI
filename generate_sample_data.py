"""Generate a realistic, deterministic sample workspace for demos, tests and evaluation.

Writes three files into examples/data:

* items.jsonl        knowledge items and tasks across three projects
* events.jsonl       ninety days of user interaction events
* eval_queries.jsonl labelled retrieval queries

Run: python examples/generate_sample_data.py
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "data"
NOW = datetime.now(UTC).replace(microsecond=0)

USERS = {
    "amara": {
        "projects": ["p_platform"],
        "groups": ["eng", "leads"],
        "topics": ["deploy", "infra", "incident"],
    },
    "ben": {"projects": ["p_platform"], "groups": ["eng"], "topics": ["api", "database", "deploy"]},
    "chloe": {
        "projects": ["p_mobile"],
        "groups": ["eng", "leads"],
        "topics": ["mobile", "release", "design"],
    },
    "dev": {"projects": ["p_mobile"], "groups": ["eng"], "topics": ["mobile", "api", "auth"]},
    "eve": {"projects": ["p_data"], "groups": ["data", "leads"], "topics": ["data", "privacy", "database"]},
    "femi": {"projects": ["p_data"], "groups": ["data"], "topics": ["data", "monitoring", "cost"]},
    "grace": {
        "projects": ["p_platform", "p_data"],
        "groups": ["security"],
        "topics": ["security", "auth", "privacy"],
    },
    "hugo": {
        "projects": ["p_platform", "p_mobile", "p_data"],
        "groups": ["product"],
        "topics": ["process", "planning", "release"],
    },
    "ivy": {"projects": ["p_mobile"], "groups": ["design"], "topics": ["design", "accessibility", "mobile"]},
    "jack": {"projects": ["p_platform"], "groups": ["eng"], "topics": ["onboarding", "deploy", "api"]},
}

DOCS = [
    (
        "kb_deploy_runbook",
        "Production deployment runbook",
        "p_platform",
        "amara",
        ["deploy", "infra"],
        "wiki_page",
        """# Before you deploy
Confirm the release branch passed the full CI suite and that the change has an approved pull request. Check the deployment calendar for freezes; no deployments run on Friday after 15:00 UK time.

# Deploying
Trigger the release pipeline from the main branch. The pipeline builds a container image, runs database migrations in a separate job, and then performs a canary rollout to 5 percent of traffic for 15 minutes.

# Verifying the canary
Watch the error rate and p95 latency on the service dashboard. If the error rate rises above 1 percent or latency doubles, stop the rollout.

# Rollback
Run the rollback job in the release pipeline and select the previous image tag. Rollback takes about four minutes. Database migrations are forward compatible, so they are never rolled back automatically.""",
    ),
    (
        "kb_incident_response",
        "Incident response process",
        "p_platform",
        "amara",
        ["incident", "oncall"],
        "document",
        """# Severity levels
SEV1 means a full outage or data loss, SEV2 a major feature unavailable, SEV3 a degraded experience with a workaround.

# Declaring an incident
Anyone can declare an incident in the incidents channel using the incident bot. The on call engineer becomes incident commander until they hand over.

# Communication
For SEV1 and SEV2, post a status update every 30 minutes and update the public status page within 15 minutes of declaration.

# After the incident
A blameless postmortem is due within five working days and must list action items with owners.""",
    ),
    (
        "kb_oncall",
        "On call rotation and handover",
        "p_platform",
        "amara",
        ["oncall", "incident"],
        "wiki_page",
        """# Rotation
The platform rotation is weekly and changes every Monday at 10:00. Primary and secondary engineers are scheduled in the paging tool.

# Handover
At handover, the outgoing engineer summarises open alerts, ongoing incidents and any risky deployments in the handover thread.

# Compensation
On call engineers receive a weekly allowance and time off in lieu for any page between 22:00 and 07:00.""",
    ),
    (
        "kb_api_rate_limits",
        "Public API rate limiting",
        "p_platform",
        "ben",
        ["api"],
        "document",
        """# Limits
The public API allows 600 requests per minute per API key and 20 concurrent requests. Bulk endpoints count as ten requests.

# Responses
When the limit is exceeded the API returns status 429 with a Retry After header in seconds. Clients must back off exponentially with jitter.

# Raising limits
Enterprise customers can request higher limits through their account manager; the platform team approves increases above 3000 requests per minute.""",
    ),
    (
        "kb_db_migrations",
        "Writing safe Postgres migrations",
        "p_platform",
        "ben",
        ["database"],
        "wiki_page",
        """# Principles
Migrations must be backward compatible with the currently deployed application version. Never rename or drop a column in the same release that stops using it.

# Large tables
Adding an index to a table with more than one million rows must use CREATE INDEX CONCURRENTLY. Backfills must run in batches of at most 10000 rows with a pause between batches.

# Review
Every migration needs a review from a member of the database guild and a tested rollback plan.""",
    ),
    (
        "kb_feature_flags",
        "Feature flag guidelines",
        "p_platform",
        "ben",
        ["deploy", "release"],
        "document",
        """# When to use a flag
Use a feature flag for any user facing change that is risky, needs a staged rollout, or must be released independently of deployment.

# Lifecycle
Flags must have an owner and an expiry date. Remove a flag within two sprints of reaching 100 percent rollout.

# Naming
Flag names use the pattern team_feature_purpose, for example mobile_checkout_new_flow.""",
    ),
    (
        "kb_oauth",
        "Authentication and OAuth token handling",
        "p_platform",
        "grace",
        ["auth", "security"],
        "document",
        """# Tokens
Access tokens expire after 15 minutes and refresh tokens after 30 days. Refresh tokens are rotated on every use, and reuse of an old refresh token revokes the whole session.

# Common errors
A 401 error with the code token_expired means the client must refresh. A 401 with invalid_grant on refresh means the session was revoked and the user must sign in again.

# Mobile clients
Mobile apps store refresh tokens in the Keychain on iOS and the encrypted Keystore on Android, never in plain preferences.""",
    ),
    (
        "kb_password_policy",
        "Password and MFA policy",
        None,
        "grace",
        ["security", "auth"],
        "document",
        """# Passwords
Passwords must be at least 14 characters. We check new passwords against known breach lists and do not force periodic rotation.

# Multi factor authentication
MFA is mandatory for all staff accounts and for admin roles in customer workspaces. Hardware security keys are preferred over authenticator apps.""",
    ),
    (
        "kb_security_incident",
        "Handling a suspected security incident",
        "p_platform",
        "grace",
        ["security", "incident"],
        "wiki_page",
        """# First steps
Do not attempt to investigate alone. Page the security on call immediately and preserve logs; do not restart affected hosts.

# Data breaches
If personal data may be affected, the security lead informs the data protection officer within 24 hours so that the 72 hour regulatory notification deadline can be met.""",
    ),
    (
        "kb_mobile_release",
        "Mobile app release checklist",
        "p_mobile",
        "chloe",
        ["mobile", "release"],
        "wiki_page",
        """# Release train
Mobile releases ship every two weeks on Tuesday. The release branch is cut on the Thursday before.

# Checklist
Update the version and build numbers, run the regression suite on the device farm, update store screenshots if the UI changed, and write release notes for both stores.

# Phased rollout
Android uses a staged rollout starting at 10 percent. iOS uses phased release over seven days. Halt the rollout if the crash free rate drops below 99.5 percent.""",
    ),
    (
        "kb_ios_signing",
        "iOS code signing and certificates",
        "p_mobile",
        "dev",
        ["mobile", "release"],
        "document",
        """# Certificates
Distribution certificates are managed with fastlane match and stored in the encrypted certificates repository. Never create certificates manually in the developer portal.

# Expiry
Certificates expire yearly; the mobile lead renews them in September. A build failing with a code signing error usually means the provisioning profile is out of date; run match in readonly mode to refresh it.""",
    ),
    (
        "kb_push_notifications",
        "Push notification service",
        "p_mobile",
        "dev",
        ["mobile", "api"],
        "document",
        """# Architecture
Notifications are sent through the notification service, which fans out to APNs for iOS and FCM for Android.

# Limits and quiet hours
Users receive at most five marketing notifications per week. Quiet hours run from 21:00 to 08:00 in the user's local time zone, except for security alerts.""",
    ),
    (
        "kb_design_system",
        "Design system usage",
        "p_mobile",
        "ivy",
        ["design"],
        "wiki_page",
        """# Components
Always use components from the shared design system library. If a component is missing, raise a request with the design team rather than building a one off.

# Tokens
Colours, spacing and typography come from design tokens. Hard coded hex values fail the lint check.""",
    ),
    (
        "kb_accessibility",
        "Accessibility standards",
        None,
        "ivy",
        ["design", "accessibility"],
        "document",
        """# Target
All new screens must meet WCAG 2.2 level AA. Text contrast must be at least 4.5 to 1.

# Testing
Test every new screen with VoiceOver on iOS and TalkBack on Android before release. Touch targets must be at least 44 by 44 points.""",
    ),
    (
        "kb_data_pipeline",
        "Data pipeline architecture",
        "p_data",
        "eve",
        ["data"],
        "document",
        """# Overview
Events flow from the product into Kafka, are landed in the data lake hourly, and are transformed with dbt into the warehouse. Airflow orchestrates every job.

# Freshness
Core dashboards refresh hourly. The finance models refresh daily at 06:00 after the billing export lands.""",
    ),
    (
        "kb_dbt_conventions",
        "dbt modelling conventions",
        "p_data",
        "eve",
        ["data"],
        "wiki_page",
        """# Layers
Use staging models for light cleaning of a single source, intermediate models for joins and business logic, and marts for consumer facing tables.

# Tests
Every primary key needs unique and not null tests. Every model needs a description and an owner in its schema file.""",
    ),
    (
        "kb_warehouse_access",
        "Requesting warehouse access",
        "p_data",
        "femi",
        ["data", "privacy"],
        "document",
        """# Access levels
Analysts get read access to marts by default. Access to raw personal data requires approval from the data protection officer and expires after 90 days.

# How to request
Raise an access request ticket with the business justification and the datasets needed. Requests are reviewed within two working days.""",
    ),
    (
        "kb_data_retention",
        "Data retention and GDPR deletion",
        "p_data",
        "eve",
        ["privacy", "data"],
        "document",
        """# Retention periods
Raw product events are kept for 13 months. Support tickets are kept for 24 months after closure. Billing records are kept for seven years for tax reasons.

# Deletion requests
Subject deletion requests must be completed within 30 days. The deletion job removes the user from the warehouse, the data lake and backups within that window.""",
    ),
    (
        "kb_cost_optimisation",
        "Cloud cost optimisation guide",
        "p_data",
        "femi",
        ["cost", "infra"],
        "document",
        """# Compute
Use spot instances for batch jobs and stop development clusters outside working hours. Right size warehouses weekly using the utilisation dashboard.

# Storage
Move data older than 90 days to the infrequent access tier. Delete unattached volumes and old snapshots monthly.""",
    ),
    (
        "kb_monitoring",
        "Monitoring and alerting standards",
        "p_platform",
        "femi",
        ["monitoring", "incident"],
        "wiki_page",
        """# Golden signals
Every service must export latency, traffic, errors and saturation metrics and have a dashboard.

# Alerts
Alerts must be actionable and link to a runbook. Page only for user impacting symptoms; everything else goes to a ticket queue.""",
    ),
    (
        "kb_code_review",
        "Code review guidelines",
        None,
        "ben",
        ["process", "api"],
        "wiki_page",
        """# Expectations
Reviews should be completed within one working day. Keep pull requests under 400 lines where possible.

# What to look for
Correctness, tests, security implications, observability and readability, in that order. Use suggestions for small fixes rather than blocking comments.""",
    ),
    (
        "kb_sprint_planning",
        "Sprint planning process",
        None,
        "hugo",
        ["process", "planning"],
        "document",
        """# Cadence
Sprints are two weeks long and start on Wednesday. Planning takes place on the first day and the review and retrospective on the last day.

# Capacity
Plan to 80 percent of team capacity to leave room for support work and incidents. Every ticket entering a sprint needs acceptance criteria and an estimate.""",
    ),
    (
        "kb_onboarding",
        "Engineering onboarding guide",
        None,
        "amara",
        ["onboarding"],
        "wiki_page",
        """# First week
Set up your laptop with the bootstrap script, request access to the code host and the paging tool, and pair with your onboarding buddy.

# First month
Ship a small change to production in your first two weeks, shadow an on call shift, and read the deployment runbook and the incident response process.""",
    ),
    (
        "dec_queue_choice",
        "Decision: Kafka for event streaming",
        "p_data",
        "eve",
        ["data", "infra"],
        "decision",
        """# Context
We needed a durable event backbone for product analytics and service integration.

# Decision
We chose Kafka over RabbitMQ because of replay, partitioned ordering and the ecosystem of connectors to the data lake.

# Consequences
The platform team operates the managed Kafka cluster. Retention is seven days on the cluster with long term storage in the lake.""",
    ),
    (
        "dec_okrs_q4",
        "Q4 OKRs",
        None,
        "hugo",
        ["planning"],
        "decision",
        """# Objectives
Improve reliability: reach 99.95 percent API availability. Grow mobile engagement: raise weekly active users by 15 percent. Make data trustworthy: all core dashboards meet freshness targets.""",
    ),
    (
        "kb_escalation",
        "Customer escalation process",
        None,
        "hugo",
        ["process", "incident"],
        "document",
        """# When to escalate
Escalate when an enterprise customer reports a production blocking issue or when a support ticket breaches its response target.

# How
Support raises the escalation in the escalations channel, and the product lead for the area assigns an engineering owner within two working hours.""",
    ),
    (
        "kb_secret_rotation",
        "Rotating secrets and API keys",
        "p_platform",
        "grace",
        ["security", "infra"],
        "wiki_page",
        """# Schedule
Service credentials are rotated every 90 days automatically by the secrets manager.

# Leaked secrets
If a secret is committed to a repository, rotate it immediately, then remove it from history. Treat the secret as compromised even if the repository is private.""",
    ),
]

TASKS = [
    (
        "t_canary_alerts",
        "Add automatic canary abort on error spike",
        "p_platform",
        "amara",
        ["amara"],
        ["deploy", "monitoring"],
        "in_progress",
        3,
        5,
        ["kb_deploy_runbook", "kb_monitoring"],
    ),
    (
        "t_rate_limit_headers",
        "Return rate limit headers on every API response",
        "p_platform",
        "ben",
        ["ben"],
        ["api"],
        "open",
        2,
        12,
        ["kb_api_rate_limits"],
    ),
    (
        "t_index_events_table",
        "Add concurrent index on events.created_at",
        "p_platform",
        "ben",
        [],
        ["database"],
        "open",
        3,
        4,
        ["kb_db_migrations"],
    ),
    (
        "t_flag_cleanup",
        "Remove stale feature flags from checkout",
        "p_platform",
        "ben",
        [],
        ["release"],
        "open",
        1,
        30,
        ["kb_feature_flags"],
    ),
    (
        "t_refresh_token_bug",
        "Fix refresh token reuse logging out Android users",
        "p_mobile",
        "dev",
        ["dev"],
        ["auth", "mobile"],
        "in_progress",
        4,
        2,
        ["kb_oauth"],
    ),
    (
        "t_ios_cert_renewal",
        "Renew iOS distribution certificate",
        "p_mobile",
        "chloe",
        [],
        ["mobile", "release"],
        "open",
        3,
        9,
        ["kb_ios_signing"],
    ),
    (
        "t_push_quiet_hours",
        "Respect quiet hours for push digest",
        "p_mobile",
        "chloe",
        [],
        ["mobile", "api"],
        "open",
        2,
        20,
        ["kb_push_notifications"],
    ),
    (
        "t_a11y_checkout",
        "Accessibility audit of the checkout screens",
        "p_mobile",
        "ivy",
        ["ivy"],
        ["accessibility", "design"],
        "open",
        2,
        15,
        ["kb_accessibility", "kb_design_system"],
    ),
    (
        "t_dbt_tests",
        "Add uniqueness tests to finance marts",
        "p_data",
        "eve",
        [],
        ["data"],
        "open",
        2,
        10,
        ["kb_dbt_conventions"],
    ),
    (
        "t_gdpr_backups",
        "Extend deletion job to cover backups",
        "p_data",
        "eve",
        ["eve"],
        ["privacy", "data"],
        "blocked",
        4,
        6,
        ["kb_data_retention"],
    ),
    (
        "t_spot_batch",
        "Move nightly batch jobs to spot instances",
        "p_data",
        "femi",
        [],
        ["cost", "infra"],
        "open",
        1,
        25,
        ["kb_cost_optimisation"],
    ),
    (
        "t_freshness_alerts",
        "Alert when core dashboards miss freshness targets",
        "p_data",
        "femi",
        ["femi"],
        ["monitoring", "data"],
        "open",
        3,
        8,
        ["kb_data_pipeline", "kb_monitoring"],
    ),
    (
        "t_secret_scan",
        "Enable secret scanning on all repositories",
        "p_platform",
        "grace",
        [],
        ["security"],
        "open",
        3,
        7,
        ["kb_secret_rotation"],
    ),
    (
        "t_mfa_admins",
        "Enforce hardware keys for workspace admins",
        "p_platform",
        "grace",
        ["grace"],
        ["security", "auth"],
        "open",
        2,
        21,
        ["kb_password_policy"],
    ),
    (
        "t_postmortem_template",
        "Update postmortem template with action item owners",
        "p_platform",
        "amara",
        [],
        ["incident", "process"],
        "open",
        1,
        18,
        ["kb_incident_response"],
    ),
    (
        "t_onboarding_bootstrap",
        "Fix bootstrap script on new laptops",
        "p_platform",
        "jack",
        ["jack"],
        ["onboarding"],
        "open",
        2,
        6,
        ["kb_onboarding"],
    ),
    (
        "t_android_staged",
        "Automate Android staged rollout halt on crash rate",
        "p_mobile",
        "chloe",
        [],
        ["mobile", "release"],
        "open",
        3,
        11,
        ["kb_mobile_release"],
    ),
    (
        "t_old_rollback",
        "Document rollback for the payments service",
        "p_platform",
        "amara",
        ["ben"],
        ["deploy"],
        "done",
        2,
        None,
        ["kb_deploy_runbook"],
    ),
    (
        "t_old_token_expiry",
        "Shorten access token lifetime to 15 minutes",
        "p_platform",
        "grace",
        ["grace"],
        ["auth"],
        "done",
        3,
        None,
        ["kb_oauth"],
    ),
    (
        "t_old_kafka_retention",
        "Set Kafka retention to seven days",
        "p_data",
        "eve",
        ["femi"],
        ["data", "infra"],
        "done",
        2,
        None,
        ["dec_queue_choice"],
    ),
]

TOPIC_OF = {}


def build_items() -> list[dict]:
    items: list[dict] = []
    for i, (iid, title, project, author, tags, kind, body) in enumerate(DOCS):
        created = NOW - timedelta(days=120 - i * 3)
        items.append(
            {
                "id": iid,
                "kind": kind,
                "title": title,
                "body": body,
                "project_id": project,
                "author_id": author,
                "tags": tags,
                "url": f"https://app.example.com/{kind}s/{iid}",
                "created_at": created.isoformat(),
                "updated_at": (created + timedelta(days=10)).isoformat(),
                "acl": {"visibility": "tenant" if project is None else "project"},
            }
        )
        TOPIC_OF[iid] = tags
    items.append(
        {
            "id": "kb_salary_bands",
            "kind": "document",
            "title": "Engineering salary bands",
            "body": "Confidential compensation bands for engineering levels, reviewed every April.",
            "author_id": "hugo",
            "tags": ["hr"],
            "acl": {"visibility": "restricted", "allowed_group_ids": ["leads"]},
        }
    )
    for i, (iid, title, project, author, assignees, tags, status, priority, due_days, links) in enumerate(
        TASKS
    ):
        created = NOW - timedelta(days=40 - i)
        body = (
            f"{title}. Context and acceptance criteria are described in the linked documents. "
            f"This work relates to {', '.join(tags)}."
        )
        items.append(
            {
                "id": iid,
                "kind": "task",
                "title": title,
                "body": body,
                "project_id": project,
                "author_id": author,
                "assignee_ids": assignees,
                "tags": tags,
                "status": status,
                "priority": priority,
                "due_at": (NOW + timedelta(days=due_days)).isoformat() if due_days is not None else None,
                "linked_item_ids": links,
                "url": f"https://app.example.com/tasks/{iid}",
                "created_at": created.isoformat(),
                "updated_at": (created + timedelta(days=2)).isoformat(),
                "acl": {"visibility": "project"},
            }
        )
        TOPIC_OF[iid] = tags
    return items


def build_events(items: list[dict]) -> list[dict]:
    rng = random.Random(7)
    by_topic: dict[str, list[str]] = {}
    for item in items:
        for tag in item.get("tags", []):
            by_topic.setdefault(tag, []).append(item["id"])
    project_of = {i["id"]: i.get("project_id") for i in items}
    events = []
    types = ["view"] * 6 + ["open"] * 2 + ["comment", "edit", "bookmark", "upvote"]
    n = 0
    for user, profile in USERS.items():
        topics = profile["topics"]
        for day in range(90, 0, -1):
            for _ in range(rng.randint(0, 3)):
                topic = rng.choice(topics) if rng.random() < 0.85 else rng.choice(list(by_topic))
                candidates = [
                    c for c in by_topic.get(topic, []) if project_of.get(c) in (None, *profile["projects"])
                ]
                if not candidates:
                    continue
                item_id = rng.choice(candidates)
                ts = NOW - timedelta(days=day, minutes=rng.randint(0, 600))
                n += 1
                events.append(
                    {
                        "event_id": f"evt_{n:05d}",
                        "type": "interaction.recorded",
                        "occurred_at": ts.isoformat(),
                        "data": {
                            "user_id": user,
                            "item_id": item_id,
                            "type": rng.choice(types),
                            "project_id": project_of.get(item_id),
                        },
                    }
                )
    events.append(
        {
            "event_id": "evt_dismiss_1",
            "type": "interaction.recorded",
            "occurred_at": (NOW - timedelta(days=1)).isoformat(),
            "data": {"user_id": "jack", "item_id": "kb_feature_flags", "type": "dismiss"},
        }
    )
    return events


EVAL_QUERIES = [
    ("how do I roll back a bad deployment", ["kb_deploy_runbook"]),
    ("what happens when a refresh token is reused", ["kb_oauth"]),
    ("API returns 429 too many requests", ["kb_api_rate_limits"]),
    ("adding an index to a huge table", ["kb_db_migrations"]),
    ("when is the mobile release branch cut", ["kb_mobile_release"]),
    ("code signing error provisioning profile", ["kb_ios_signing"]),
    ("how long do we keep billing records", ["kb_data_retention"]),
    ("who approves access to raw personal data", ["kb_warehouse_access"]),
    ("why did we choose kafka", ["dec_queue_choice"]),
    ("status page update during an outage", ["kb_incident_response"]),
    ("minimum contrast ratio for text", ["kb_accessibility"]),
    ("secret committed to git by mistake", ["kb_secret_rotation"]),
    ("reduce warehouse costs", ["kb_cost_optimisation"]),
    ("quiet hours for notifications", ["kb_push_notifications"]),
    ("how big should a pull request be", ["kb_code_review"]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    events = build_events(items)
    with (OUT / "items.jsonl").open("w") as f:
        for item in items:
            f.write(json.dumps({k: v for k, v in item.items() if v is not None}) + "\n")
    with (OUT / "events.jsonl").open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    with (OUT / "eval_queries.jsonl").open("w") as f:
        for query, relevant in EVAL_QUERIES:
            f.write(
                json.dumps(
                    {
                        "query": query,
                        "relevant_item_ids": relevant,
                        "principal": {"user_id": "evaluator", "is_admin": True},
                    }
                )
                + "\n"
            )
    print(f"items={len(items)} events={len(events)} queries={len(EVAL_QUERIES)} written to {OUT}")


if __name__ == "__main__":
    main()
