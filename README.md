# Email Triage OpenEnv

A real-world AI environment simulating email triage and support escalation workflows.

## Motivation
Email triage is a common enterprise workflow task performed by support teams, IT operations, and customer service agents.

This environment simulates:
- classification
- prioritization
- escalation

## Tasks
### Easy
Urgent billing issue

### Medium
Meeting scheduling request

### Hard
Multi-intent technical complaint

## Reward
Score range: 0.0 to 1.0

- category = 0.3
- priority = 0.3
- escalation = 0.2
- multi-step completion = 0.2

## Run
```bash
python3 inference.py
```

## Docker
```bash
docker build -t email-env .
docker run email-env
```