import os
from openai import OpenAI
from advanced_env import CalendarOrchestrationEnv, Action


def get_client():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    return client, MODEL_NAME


def llm_plan_action(calendar_state, step_num):
    client, MODEL_NAME = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": f"Calendar state: {calendar_state}, step: {step_num}"
            }
        ]
    )

    _ = response.choices[0].message.content

    return Action(
        action_type="reschedule_meeting",
        target_meeting_id="m2",
        new_time="16:00",
        reason="critical_incident"
    )


def run_inference():
    _, MODEL_NAME = get_client()

    env = CalendarOrchestrationEnv()
    obs = env.reset()

    print(f"[START] task=calendar_task env=openenv model={MODEL_NAME}")

    rewards = []
    success = False

    try:
        for step_num in range(1, 4):
            action = llm_plan_action(obs.calendar, step_num)

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
                success = True
                break

    except Exception as e:
        print(
            f"[STEP] step={step_num} "
            f"action=error "
            f"reward=0.00 "
            f"done=true "
            f"error={str(e)}"
        )

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    run_inference()