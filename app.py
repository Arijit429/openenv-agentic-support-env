from flask import Flask
from inference import run_task
from email_env import Action

app = Flask(__name__)

@app.route("/")
def home():
    return {
        "status": "running",
        "message": "OpenEnv Agentic Support Environment is live"
    }

@app.route("/reset")
def reset():
    return {
        "status": "reset_ok"
    }

@app.route("/run")
def run():
    run_task(
        1,
        [
            Action(action_type="analyze_ticket", resolution_note=""),
            Action(action_type="process_refund", resolution_note=""),
            Action(action_type="close_ticket", resolution_note="")
        ]
    )
    return {"status": "task_executed"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)