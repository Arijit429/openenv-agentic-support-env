---
title: OpenEnv Agentic Support Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv Agentic Support Environment

A real-world **agentic AI workflow environment** for autonomous support ticket resolution.

## Motivation
This environment is designed for training and benchmarking AI agents on realistic enterprise workflows such as:
- ticket analysis
- escalation
- refund processing
- incident management

## Tasks
### Easy
Refund issue

### Medium
Technical issue escalation

### Hard
Enterprise outage response

## Reward
Reward range: 0.0 to 1.0

## Run
```bash
python3 inference.py
```

## Docker
```bash
docker build -t agent-env .
docker run agent-env
```