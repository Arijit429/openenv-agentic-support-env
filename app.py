from flask import Flask, request, jsonify
from inference import run_task
from email_env import Action, SupportAgentEnv

app = Flask(__name__)

env = SupportAgentEnv()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "OpenEnv Agentic Support Environment is live"
    })


@app.route("/reset", methods=["POST", "GET"])
def reset():
    obs = env.reset(task_id=1)

    return jsonify({
        "status": "reset_ok",
        "observation": obs.model_dump()
    })


@app.route("/state", methods=["GET"])
def state():
    return jsonify(env.state())


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json()

    action = Action(
        action_type=data.get("action_type", ""),
        resolution_note=data.get("resolution_note", "")
    )

    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    })


@app.route("/run", methods=["GET"])
def run():
    run_task(
        1,
        [
            Action(action_type="analyze_ticket", resolution_note=""),
            Action(action_type="process_refund", resolution_note=""),
            Action(action_type="close_ticket", resolution_note="")
        ]
    )

    return jsonify({"status": "task_executed"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)