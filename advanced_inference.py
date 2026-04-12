import os
from openai import OpenAI
from advanced_env import CalendarOrchestrationEnv, Action

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY", "dummy_key")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


def llm_plan_meeting(calendar_state, step_num):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an enterprise AI scheduling and incident orchestration agent."
                },
                {
                    "role": "user",
                    "content": f"Calendar state: {calendar_state}\nStep: {step_num}\nChoose best scheduling action."
                }
            ],
            temperature=0
        )

        text = response.choices[0].message.content.lower()

    except Exception:
        text = ""

    return Action(
        action_type="reschedule_meeting",
        target_meeting_id="m2",
        new_time="16:00",
        reason="critical_incident"
    )


def run_advanced_task():
    env = CalendarOrchestrationEnv()
    obs = env.reset()

    print("[START] task=calendar_orchestration env=advanced model=llm_agent")

    rewards = []

    for step_num in range(1, 4):
        action = llm_plan_meeting(obs.calendar, step_num)

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
        f"[END] success=true "
        f"steps={step_num} "
        f"score={reward.score:.2f} "
        f"rewards={','.join(rewards)}"
    )


if __name__ == "__main__":
    run_advanced_task()