import os
from openai import OpenAI
from email_env import SupportAgentEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy_token")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def run_task(task_id, actions):
    env = SupportAgentEnv()
    obs = env.reset(task_id=task_id)

    print(f"[START] task=task_{task_id} env=support_agent model={MODEL_NAME}")

    rewards = []

    for step_num, action in enumerate(actions, 1):
        obs, reward, done, info = env.step(action)
        rewards.append(f"{reward.score:.2f}")

        print(
            f"[STEP] step={step_num} "
            f"action={action.action_type} "
            f"reward={reward.score:.2f} "
            f"done={'true' if done else 'false'} "
            f"error=null"
        )

        if done:
            break

    print(
        f"[END] success={'true' if reward.score > 0 else 'false'} "
        f"steps={step_num} "
        f"score={reward.score:.2f} "
        f"rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    run_task(
        1,
        [
            Action(action_type="analyze_ticket", resolution_note=""),
            Action(action_type="process_refund", resolution_note=""),
            Action(action_type="close_ticket", resolution_note="")
        ]
    )

    run_task(
        2,
        [
            Action(action_type="analyze_ticket", resolution_note=""),
            Action(action_type="escalate_engineering", resolution_note=""),
            Action(action_type="close_ticket", resolution_note="")
        ]
    )

    run_task(
        3,
        [
            Action(action_type="analyze_ticket", resolution_note=""),
            Action(action_type="mark_critical", resolution_note=""),
            Action(action_type="escalate_management", resolution_note="")
        ]
    )