from dataclasses import dataclass
from typing import Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI()
# как идея создать ещё переменную хранящую общее кол-во записей
# делать это в классе и иметь setter и getter

click_count: int = 0
# offer_id: list[click_id:int]
offer_clicks = {}
# click_id:int : reward:int
click_reward = {}


@app.put("/feedback/")
def feedback(click_id: int, reward: float) -> dict:
    """Get feedback for particular click"""
    # Response body consists of click ID
    # and accepted click status (True/False)
    if click_id not in click_reward:
        try:
            offer_id = [offer_id for offer_id, click_ids in offer_clicks.items() if click_id in click_ids][0]
            offer_id = int(offer_id)

            is_conversion = reward != 0
            click_reward[click_id] = reward

        except IndexError as e:
            # В случае, если offer_id пустой (нет совпадений)
            raise HTTPException(status_code=404, detail=f"Offer not found, {e.__traceback__}") from e
    else:
        raise HTTPException(status_code=404, detail="This click already has a record of reward")

    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": reward
    }
    return response


@app.get("/offer_ids/{offer_id}/stats/")
def stats(offer_id: int) -> dict:
    """Return offer's statistics"""
    if offer_id not in offer_clicks:
        print("-------------- offer_id is none")
        response = {
            "offer_id": offer_id,
            "clicks": None,
            "conversions": None,
            "reward": None,
            "cr": None,
            "rpc": None,
        }
    else:
        print(offer_clicks[offer_id])
        clicks = offer_clicks[offer_id]
        rewarding_clicks = [click_reward[click_id] for click_id in clicks if click_id in click_reward]
        conversions = len(clicks) - rewarding_clicks.count(0)
        reward = sum(rewarding_clicks)

        print(click_reward)
        response = {
            "offer_id": offer_id,
            "clicks": len(clicks),
            "conversions": conversions,
            "reward": reward,
            "cr": conversions / len(clicks) if conversions is not None else None,
            "rpc": reward / len(clicks) if reward is not None else None,
        }

    return response


@app.get("/sample/")
def sample(click_id: int, offer_ids: str) -> dict:
    """Greedy sampling"""
    # Parse offer IDs
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Sample top offer ID
    if click_count <= 100:
        offer_id = int(np.random.choice(offers_ids))
    else:
        # реализовать алгоритм максимизации RPC
        offer_id = int(np.random.choice(offers_ids))
    # Prepare response
    add_click_to_offer(click_id, offer_id)
    response = {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": "random",
    }

    return response


def add_click_to_offer(click_id, offer_id):
    """

    :param click_id:
    :param offer_id:
    :return:
    """
    # добавить обработку исключения, если отдаётся уже записанный click
    global click_count
    global offer_clicks
    if check_number_in_values(offer_clicks, click_id):
        raise HTTPException(status_code=404, detail="This click_id is already registered")
    if offer_id not in offer_clicks:
        offer_clicks[offer_id] = []
    offer_clicks[offer_id].append(click_id)
    click_count += 1
    print(offer_clicks)
    print(click_reward)

def check_number_in_values(dictionary, number):
    for _, value in dictionary.items():
        if number in value:
            return True
    return False

def main() -> None:
    """Run application"""
    uvicorn.run("app:app", host="localhost", reload=True)


if __name__ == "__main__":
    main()
